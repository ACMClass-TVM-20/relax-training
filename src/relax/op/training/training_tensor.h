/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#ifndef TVM_RELAX_OP_TRAINING_TRAINING_TENSOR_H_
#define TVM_RELAX_OP_TRAINING_TRAINING_TENSOR_H_

#include "training.h"

namespace tvm {
namespace relax {

Optional<Expr> InferShapeCollapseSumLike(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 2) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Matmul op should have 2 arguments");
  }
  return call->args[1]->shape();
}

Type InferTypeCollapseSumLike(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 2) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "Matmul op should have 2 arguments");
  }
  return call->args[1]->checked_type();
}

Optional<Expr> InferShapeOnesLike(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 1) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "ones-like op should have 1 arguments");
  }
  return call->args[0];
}

Type InferTypeOnesLike(const Call& call, DiagnosticContext diag_ctx) {
  if (call->args.size() != 1) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "ones-like op should have 1 arguments");
  }
  Type type0 = call->args[0]->checked_type();
  auto* t0 = type0.as<ShapeTypeNode>();
  if (!t0) {
    diag_ctx.EmitFatal(Diagnostic::Error(call->span) << "ones-like op should take a ShapeExpr");
  }
  return DynTensorType(call->args[0].as<ShapeExprNode>()->values.size(), DataType::Float(32));
}

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_TRAINING_TRAINING_TENSOR_H_
