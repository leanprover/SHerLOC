"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}, {mhlo.sharding = ""}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x2xf32>, tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<?x4xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg4: tensor<i64>, %arg5: tensor<?x2xf32>, %arg6: tensor<2x3xf32>, %arg7: tensor<3x4xf32>):
    %2 = "func.call"(%arg4, %arg5, %arg6, %arg7) <{callee = @_einsum}> : (tensor<i64>, tensor<?x2xf32>, tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<?x4xf32>
    "func.return"(%2) : (tensor<?x4xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<i64>, tensor<?x2xf32>, tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<?x4xf32>, sym_name = "_einsum", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x2xf32>, %arg2: tensor<2x3xf32>, %arg3: tensor<3x4xf32>):
    %0 = "stablehlo.dot_general"(%arg3, %arg2) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [1]>}> : (tensor<3x4xf32>, tensor<2x3xf32>) -> tensor<4x2xf32>
    %1 = "stablehlo.dot_general"(%arg1, %0) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>}> : (tensor<?x2xf32>, tensor<4x2xf32>) -> tensor<?x4xf32>
    "func.return"(%1) : (tensor<?x4xf32>) -> ()
  }) : () -> ()
}) : () -> ()

