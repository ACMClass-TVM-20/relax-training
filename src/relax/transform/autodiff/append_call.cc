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
 * \file src/relax/transform/autodiff/append_call.cc
 * \brief A pass which will append a single var binding with a call in the only dataflow block of specified function.
 *
 *  TODO(chaofanlin): API
 */

#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/relay/op_attr_types.h>

namespace tvm {
namespace relax {

class AppendCallMutator : public ExprMutator {
 public:
  explicit AppendCallMutator(IRModule mod, const Op& op_, const Var& out_, const Array<Var>& args_)
      : ExprMutator(mod), op(op_), out(out_), args(args_) {}

  Function FuncTransform(const FunctionNode* node) {
    ICHECK(node->body->IsInstance<SeqExprNode>());

    Array<Var> new_params, bindings_var;
    for (Var param : node->params) {
      Var new_param = Var(param->vid, NullOpt, param->checked_type_, param->span);
      UpdateShape(new_param, param->shape_);
      this->var_remap_[param->vid] = new_param;
      new_params.push_back(new_param);
    }

    Array<Expr> args_expr; // convert Array<Var> to Array<Expr>
    for (auto arg: args) {
      // if arg not in current input and not the body, add it into parms
      if (std::find(node->params.begin(), node->params.end(), arg) == node->params.end() &&
          arg != node->body.as<SeqExprNode>()->body) {
        Var new_param = Var(arg->vid, NullOpt, arg->checked_type_, arg->span);
        UpdateShape(new_param, arg->shape_);
        this->var_remap_[arg->vid] = new_param;
        new_params.push_back(new_param);
      }
      args_expr.push_back(arg);
    }

    Expr new_body = VisitWithNewScope(node->body);

    const SeqExprNode* seq_expr = new_body.as<SeqExprNode>();
    // only a single dataflow block
    ICHECK(seq_expr->blocks.size() == 1);
    ICHECK(seq_expr->blocks[0]->IsInstance<DataflowBlockNode>());
    const DataflowBlockNode* block = seq_expr->blocks[0].as<DataflowBlockNode>();

    builder_->BeginDataflowBlock();
    for (const auto& binding: block->bindings) {
      VisitBinding(binding);
    }
    builder_->EmitOutput(VarBinding(out, Call(op, args_expr)));
    Expr final_body = builder_->Normalize(VisitWithNewScope(SeqExpr({builder_->EndBlock()}, out)));

    return Function(new_params, final_body, out->checked_type_, out->shape(), node->attrs);
  }

 private:
  Op op;
  Var out;
  Array<Var> args;
};

IRModule AppendCall(IRModule m, const GlobalVar &var, const Op& op, const Var& out, const Array<Var>& args) {
  BaseFunc base_func = m->Lookup(var);
  if (auto* n = base_func.as<FunctionNode>()) {
    auto f_before = GetRef<Function>(n);
    IRModuleNode* new_module_node = m.CopyOnWrite();
    auto new_module = GetRef<IRModule>(new_module_node);
    auto mutator = AppendCallMutator(new_module, op, out, args);
    auto adjoint_var = GlobalVar(var->name_hint + "1");
    Function f_after = mutator.FuncTransform(f_before.as<FunctionNode>());
    f_after = WithAttr(f_after, tvm::attr::kGlobalSymbol, adjoint_var->name_hint);
    new_module->Add(adjoint_var, f_after);
    return new_module;
  } else {
    LOG(FATAL) << "relax function " << var->name_hint << " not found";
    return m;
  }
}

namespace transform {

Pass AppendCall(GlobalVar func, Op op, Var out, Array<Var> args) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule mod, PassContext pc) {
        return relax::AppendCall(mod, func, op, out, args);
      };
  return CreateModulePass(/*pass_function=*/pass_func,
                          /*opt_level=*/0,
                          /*pass_name=*/"AppendCall",
                          /*required=*/{});
}

TVM_REGISTER_GLOBAL("relax.transform.AppendCall").set_body_typed(AppendCall);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
