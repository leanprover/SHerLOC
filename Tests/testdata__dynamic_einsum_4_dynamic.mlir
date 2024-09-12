"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<3x?xf32>, tensor<?x5xf32>) -> tensor<3x5xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg3: tensor<i64>, %arg4: tensor<3x?xf32>, %arg5: tensor<?x5xf32>):
    %1 = "stablehlo.concatenate"(%arg4, %arg4) <{dimension = 1 : i64}> : (tensor<3x?xf32>, tensor<3x?xf32>) -> tensor<3x?xf32>
    %2 = "stablehlo.concatenate"(%arg5, %arg5) <{dimension = 0 : i64}> : (tensor<?x5xf32>, tensor<?x5xf32>) -> tensor<?x5xf32>
    %3 = "func.call"(%arg3, %1, %2) <{callee = @_einsum}> : (tensor<i64>, tensor<3x?xf32>, tensor<?x5xf32>) -> tensor<3x5xf32>
    "func.return"(%3) : (tensor<3x5xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<i64>, tensor<3x?xf32>, tensor<?x5xf32>) -> tensor<3x5xf32>, sym_name = "_einsum", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<3x?xf32>, %arg2: tensor<?x5xf32>):
    %0 = "stablehlo.dot_general"(%arg1, %arg2) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>}> : (tensor<3x?xf32>, tensor<?x5xf32>) -> tensor<3x5xf32>
    "func.return"(%0) : (tensor<3x5xf32>) -> ()
  }) : () -> ()
}) : () -> ()

