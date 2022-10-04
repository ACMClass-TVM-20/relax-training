/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
/*!
 * \file src/relax/transform/autodiff/simple_ad.cc
 * \brief A simple reverse-mode auto differentiation.
 */

#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/relay/op_attr_types.h>

#include <unordered_set>

namespace tvm {
namespace relax {

class SimpleADMutator : public ExprMutator {
  public:
    explicit SimpleADMutator(IRModule mod, const Array<String>& target_names, const Array<String>& require_grad_names):
    ExprMutator(mod), target_names_(), require_grad_names_() {
        for (const String& name: target_names) {
            target_names_.emplace(name);
        }
        for (const String& name: require_grad_names) {
            require_grad_names_.emplace(name);
        }
    }

    Expr VisitExpr_(const FunctionNode* node) override {
        ICHECK(node->body->IsInstance<SeqExprNode>());
        const SeqExprNode* seq_expr = node->body.as<SeqExprNode>();
        // only a single dataflow block
        ICHECK(seq_expr->blocks.size() == 1);
        ICHECK(seq_expr->blocks[0]->IsInstance<DataflowBlockNode>());
        const DataflowBlockNode* block = seq_expr->blocks[0].as<DataflowBlockNode>();
        
        builder_->BeginDataflowBlock();
        // copy and emit
        for (const auto& binding: block->bindings) {
            EmitBinding(binding);
        }

        for (const auto& v: node->params) {
            if (require_grad_names_.empty() || require_grad_names_.count(v->name_hint())) {
                CreateAdjointVar(v, false);
            }
            else {
                CreateAdjointVar(v, true);
            }
        }

        // reverse-mode
        for (int i = block->bindings.size()-1; i >= 0; --i) {
            if (!block->bindings[i]->IsInstance<VarBindingNode>()) continue;
            const VarBindingNode* binding = block->bindings[i].as<VarBindingNode>();
            VisitBinding_(binding);  
        }

        // handle the return
        Array<Expr> out_expr, out_shape;
        Array<Type> ret_type;
        out_expr.push_back(seq_expr->body);
        out_shape.push_back(seq_expr->body->shape());
        ret_type.push_back(node->ret_type);

        // emit the input adjoints
        for (const auto& param: node->params) {
            if (require_grad_names_.empty() || require_grad_names_.count(param->name_hint())) {
                const Var& adjoint_var = adjoint_var_map[param];
                BindAndEmit(adjoint_var, adjoint_expr_map[param]);
                out_expr.push_back(adjoint_var);
                out_shape.push_back(adjoint_var->shape());
                ret_type.push_back(adjoint_var->checked_type());
            }
        }

        return Function(node->params, SeqExpr({builder_->EndBlock()}, Tuple(out_expr)), TupleType(ret_type), node->attrs);
    }

    void VisitBinding_(const VarBindingNode* binding) override {
        CreateAdjointVar(binding->var, true);
        const Var& adjoint_var = adjoint_var_map[binding->var];

        // must be output or expr in ignored output's AST
        if (adjoint_expr_map.count(binding->var) == 0) {
            if (!target_names_.empty() && target_names_.count(binding->var->name_hint()) == 0) {
                return;
            }
            ICHECK(!binding->var->IsInstance<DataflowVarNode>()) << "not an output node";
            const Op& init_op = Op::Get("relax.ones_like");
            BindAndEmit(adjoint_var, Call(init_op, {binding->var}));
        }
        else {
            // meet a def
            BindAndEmit(adjoint_var, adjoint_expr_map[binding->var]);
        }

        // back prop.

        // case 1: assign
        // a = b
        // b_adjoint_expr += a_adjoint_var
        if (const auto* node = binding->value.as<VarNode>()) {
            AdjointExprIncre(GetRef<Var>(node), adjoint_var);
        }
        // case 2: call
        else if (const auto* node = binding->value.as<CallNode>()) {
            const Op& call_op = GetRef<Op>(node->op.as<OpNode>());
            const Array<Expr>& partials = gradient_op_map[call_op](GetRef<Call>(node), adjoint_var);
            ICHECK(partials.size() == node->args.size()) << "partials number != inputs number";
            for (size_t i = 0; i < partials.size(); ++i) {
                const VarNode* arg = node->args[i].as<VarNode>();
                ICHECK(arg != nullptr);
                AdjointExprIncre(GetRef<Var>(arg), partials[i]);
            }
        }
        else {
            LOG(FATAL) << "Unsupport: unknown binding expr" << binding->value;
        }

        // SSA. release the space
        adjoint_var_map.erase(binding->var);
        adjoint_expr_map.erase(binding->var);
    }

    void CreateAdjointVar(const Var& v, bool is_dataflow_var) {
        // has created
        if (adjoint_var_map.count(v) != 0) return;
        if (is_dataflow_var) {
            Var adjoint = DataflowVar(v->name_hint() + "_adjoint", v->shape(), v->checked_type());
			adjoint->checked_type_ = v->checked_type();
            adjoint_var_map.Set(v, adjoint);
        }
        else {
            Var adjoint = Var(v->name_hint() + "_adjoint", v->shape(), v->checked_type());
			adjoint->checked_type_ = v->checked_type();
            adjoint_var_map.Set(v, adjoint);
        }
    }

    void AdjointExprIncre(const Var& v, const Expr& increment) {
        if (adjoint_expr_map.count(v) == 0) {
            adjoint_expr_map.Set(v, increment);
        }
        else {
            const Expr& now_expr = adjoint_expr_map[v];
            const Op& add_op = Op::Get("relax.add");
            const Expr& new_expr = Call(add_op, {now_expr, increment});
            adjoint_expr_map.Set(v, new_expr);
        }
    }

    void EmitBinding(const Binding& binding) {
        if (const auto* node = binding.as<VarBindingNode>()) {
            const VarBinding& var_binding = GetRef<VarBinding>(node);
            if (var_binding->var->IsInstance<DataflowVarNode>()) {
                builder_->Emit(var_binding);
            }
            else {
                builder_->EmitOutput(var_binding);
            }
        }
        else if (const auto* node = binding.as<MatchShapeNode>()) {
            const MatchShape& match_shape = GetRef<MatchShape>(node);
            builder_->EmitMatchShape(match_shape);
        } 
        else {
            LOG(FATAL) << "TypeError: Invalid type: " << binding->GetTypeKey();
        }
    }

    void BindAndEmit(const Var& v, const Expr& e) {
        e->checked_type_ = v->checked_type();
        e->shape_ = v->shape();
        if (v->IsInstance<DataflowVarNode>()) {
            builder_->Emit(VarBinding(v, e));
        }
        else {
            builder_->EmitOutput(VarBinding(v, e));
        }
    }

  private:
    // specified sets
    std::unordered_set<String> target_names_, require_grad_names_;

    // var to its adjoints var
    Map<Var, Var> adjoint_var_map;
    // var to its adjoint expr
    Map<Var, Expr> adjoint_expr_map;

    // gop map
    const OpAttrMap<relay::FPrimalGradient> gradient_op_map = Op::GetAttrMap<relay::FPrimalGradient>("FPrimalGradient");
};

IRModule SimpleAD(IRModule m, String func_name, const Array<String>& target_names, const Array<String>& require_grad_names) {
  IRModuleNode* new_module = m.CopyOnWrite();
  auto mutator = SimpleADMutator(GetRef<IRModule>(new_module), target_names, require_grad_names);
  for (const auto& func_pr : m->functions) {
    if (const auto* relax_f = func_pr.second.as<FunctionNode>()) {
      Optional<String> gsymbol = relax_f->GetAttr<String>(tvm::attr::kGlobalSymbol);
      if (gsymbol.defined() && gsymbol.value() == func_name) {
        Function f_after = Downcast<Function>(mutator.VisitExpr(func_pr.second));
        new_module->Update(func_pr.first, f_after);
      }
    }
  }
  return GetRef<IRModule>(new_module);
}

namespace transform {

Pass SimpleAD(String func_name, Array<String> target_names, Array<String> require_grad_names) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule mod, PassContext pc) { 
        return relax::SimpleAD(mod, func_name, target_names, require_grad_names); 
      };
  return CreateModulePass(/*pass_function=*/pass_func,  //
                          /*opt_level=*/0,              //
                          /*pass_name=*/"SimpleAD",      //
                          /*required=*/{});
}

TVM_REGISTER_GLOBAL("relax.transform.SimpleAD").set_body_typed(SimpleAD);

}  // namespace transform

}
}
