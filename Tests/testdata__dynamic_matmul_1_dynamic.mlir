"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x8x4xf32>, tensor<4x5xf32>) -> tensor<?x8x5xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x8x4xf32>, %arg2: tensor<4x5xf32>):
    %0 = "stablehlo.dot_general"(%arg1, %arg2) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [0]>}> : (tensor<?x8x4xf32>, tensor<4x5xf32>) -> tensor<?x8x5xf32>
    "func.return"(%0) : (tensor<?x8x5xf32>) -> ()
  }) : () -> ()
}) : () -> ()

