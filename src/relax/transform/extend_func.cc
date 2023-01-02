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
 * \file src/relax/transform/extend_func.cc
 * \brief Extend a relax.Function in the mod by another given function.
 *
 * It will reserve the original function and let the extended function be a
 * new function in the mod, with the name of "<orig_func_name>_<ex_func_name>".
 * Here <orig_func_name> means the name of its corresponding gvar while <ex_func_name>
 * is acquired by global symbol attribute. If it is none, use `foo` as the default name.
 */

#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

namespace tvm {
namespace relax {

class ExtendFuncMutator : public ExprMutator {
 public:
  explicit ExtendFuncMutator(const IRModule& mod, const GlobalVar& gvar, const Function& ex_func)
      : ExprMutator(mod), mod_(mod), gvar_(gvar), ex_func_(ex_func) {}

  IRModule Transform() {
    auto new_module = GetRef<IRModule>(mod_.CopyOnWrite());

    auto func_before = Downcast<Function>(mod_->Lookup(gvar_));
    String ex_func_name = ex_func_->GetAttr<String>(tvm::attr::kGlobalSymbol).value_or("foo");

    auto func_after_var = GlobalVar(gvar_->name_hint + "_" + ex_func_name);
    auto func_after = Downcast<Function>(this->VisitExpr(func_before));
    new_module->Add(func_after_var, func_after);
    return new_module;
  }

  Expr VisitExpr_(const FunctionNode* func) override {
    CHECK(func->body->IsInstance<SeqExprNode>())
        << "the body of the original function is not SeqExpr.";
    CHECK(ex_func_->body->IsInstance<SeqExprNode>())
        << "the body of the ex function is not SeqExpr.";

    Array<Var> new_params;
    for (Var param : func->params) {
      Var new_param = Var(param->vid, param->shape(), param->checked_type(), param->span);
      this->var_remap_[param->vid] = new_param;
      new_params.push_back(new_param);
    }

    SeqExpr seq_expr = Downcast<SeqExpr>(func->body);

    for (BindingBlock block : seq_expr->blocks) {
      for (Binding binding : block->bindings) {
        const auto* binding_node = binding.as<VarBindingNode>();
        if (binding_node && !binding_node->var->IsInstance<DataflowVarNode>()) {
          Var new_binding_var =
              DataflowVar(binding_node->var->vid, binding_node->var->shape(),
                          binding_node->var->checked_type(), binding_node->var->span);
          this->var_remap_[binding_node->var->vid] = new_binding_var;
        }
      }
    }

    if (func->ret_type.as<TupleTypeNode>()) {
      const auto* tuple_node = seq_expr->body.as<TupleNode>();
      ICHECK(tuple_node != nullptr);
      for (Expr field : tuple_node->fields) {
        orig_rets_.push_back(this->VisitExpr(field));
      }
    } else {
      orig_rets_.push_back(this->VisitExpr(seq_expr->body));
    }

    CHECK(ex_func_->params.size() >= orig_rets_.size())
        << "The number of return values of original functions should be greater than the number of "
           "parameters of ex function";

    for (int i = 0; i < static_cast<int>(ex_func_->params.size()); ++i) {
      Var ex_param = ex_func_->params[i];
      if (i < static_cast<int>(orig_rets_.size())) {
        // TODO(chaofan): a better way to check whether new_ret_var should be dataflow
        if (const auto* var_node = orig_rets_[i].as<VarNode>()) {
          ICHECK(orig_rets_[i].as<DataflowVarNode>());
          orig_rets_var_.push_back(NullOpt);
          this->var_remap_[ex_param->vid] = GetRef<Var>(var_node);
        } else {
          Var new_ret_var = DataflowVar(/*name_hint=*/"ret_" + std::to_string(i),
                                        orig_rets_[i]->shape(), orig_rets_[i]->checked_type());
          orig_rets_var_.push_back(new_ret_var);
          this->var_remap_[ex_param->vid] = new_ret_var;
        }
      } else {
        Var new_ex_param =
            Var(ex_param->vid, ex_param->shape(), ex_param->checked_type(), ex_param->span);
        this->var_remap_[ex_param->vid] = new_ex_param;
        new_params.push_back(new_ex_param);
      }
    }

    remapped_ex_body_ = static_cast<ExprMutator>(*this).VisitExpr(ex_func_->body);

    Expr new_body = this->VisitExpr(func->body);

    return Function(new_params, new_body, Type(), RuntimeDepShape(), func->attrs);
  }

  Expr VisitExpr_(const SeqExprNode* seq_expr) override {
    Array<BindingBlock> blocks;
    for (int i = 0; i < static_cast<int>(seq_expr->blocks.size()); ++i) {
      if (i < static_cast<int>(seq_expr->blocks.size()) - 1) {
        blocks.push_back(seq_expr->blocks[i]);
      } else {
        BindingBlock new_block = this->VisitBindingBlock(seq_expr->blocks[i]);
        if (!new_block->bindings.empty()) {
          blocks.push_back(new_block);
        }
      }
    }
    this->VisitExpr(seq_expr->body);
    return SeqExpr(blocks, Downcast<SeqExpr>(remapped_ex_body_)->body);
  }

  BindingBlock VisitBindingBlock_(const DataflowBlockNode* block) override {
    builder_->BeginDataflowBlock();
    for (const auto& binding : block->bindings) {
      this->VisitBinding(binding);
    }

    ICHECK(orig_rets_var_.size() == orig_rets_.size());
    for (int i = 0; i < static_cast<int>(orig_rets_var_.size()); ++i) {
      if (orig_rets_var_[i].defined()) {
        builder_->Emit(VarBinding(orig_rets_var_[i].value(), orig_rets_[i]));
      }
    }

    ICHECK(remapped_ex_body_->IsInstance<SeqExprNode>());
    const Array<BindingBlock>& blocks = Downcast<SeqExpr>(remapped_ex_body_)->blocks;

    for (BindingBlock block : blocks) {
      for (Binding binding : block->bindings) {
        this->VisitBinding(binding);
      }
    }

    return builder_->EndBlock();
  }

  void VisitBinding_(const VarBindingNode* binding) override {
    if (this->var_remap_.count(binding->var->vid)) {
      Expr new_value = this->VisitExpr(binding->value);
      builder_->Emit(VarBinding(this->var_remap_[binding->var->vid], new_value));
      return;
    }
    ExprMutator::VisitBinding_(binding);
  }

 private:
  IRModule mod_;
  GlobalVar gvar_;
  Function ex_func_;

  Array<Optional<Var>> orig_rets_var_;
  Array<Expr> orig_rets_;

  Expr remapped_ex_body_;
};

/*!
 * \brief This is the internal function of tvm::relax::transform::ExtendFunc.
 * \param mod The module
 * \param gvar The GlobalVar of the function to be extended.
 * \param ex_func The function to be linked after the target function.
 * \return The module after transformation.
 */
IRModule ExtendFunc(const IRModule& mod, const GlobalVar& gvar, const Function& ex_func) {
  auto* func = mod->Lookup(gvar).as<FunctionNode>();
  CHECK(func) << "relax function " << gvar->name_hint << " is not found";
  return ExtendFuncMutator(mod, gvar, ex_func).Transform();
}

namespace transform {

Pass ExtendFunc(GlobalVar global_var, Function ex_func) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule mod, PassContext pc) { return relax::ExtendFunc(mod, global_var, ex_func); };
  return CreateModulePass(/*pass_function=*/pass_func,
                          /*opt_level=*/0,
                          /*pass_name=*/"ExtendFunc",
                          /*required=*/{});
}

TVM_REGISTER_GLOBAL("relax.transform.ExtendFunc").set_body_typed(ExtendFunc);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
