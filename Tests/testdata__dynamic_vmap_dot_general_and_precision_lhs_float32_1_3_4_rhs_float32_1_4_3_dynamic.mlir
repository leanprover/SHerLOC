"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x1x3x4xf32>, tensor<?x1x4x3xf32>) -> tensor<?x1xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x1x3x4xf32>, %arg2: tensor<?x1x4x3xf32>):
    %0 = "stablehlo.dot_general"(%arg1, %arg2) <{dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0, 1], rhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3, 2], rhs_contracting_dimensions = [2, 3]>, precision_config = [#stablehlo<precision HIGHEST>, #stablehlo<precision HIGHEST>]}> : (tensor<?x1x3x4xf32>, tensor<?x1x4x3xf32>) -> tensor<?x1xf32>
    "func.return"(%0) : (tensor<?x1xf32>) -> ()
  }) : () -> ()
}) : () -> ()

