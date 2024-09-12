"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x4x2x3x5xf32>, tensor<?x4x3xf32>) -> tensor<?x4x2x3x5xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x4x2x3x5xf32>, %arg2: tensor<?x4x3xf32>):
    %0 = "stablehlo.constant"() <{value = dense<[0, 4]> : tensor<2xi64>}> : () -> tensor<2xi64>
    %1 = "stablehlo.scatter"(%arg1, %0, %arg2) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1, 2], inserted_window_dims = [2, 4], scatter_dims_to_operand_dims = [2, 4]>, unique_indices = true}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %2 = "stablehlo.minimum"(%arg3, %arg4) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%2) : (tensor<f32>) -> ()
    }) : (tensor<?x4x2x3x5xf32>, tensor<2xi64>, tensor<?x4x3xf32>) -> tensor<?x4x2x3x5xf32>
    "func.return"(%1) : (tensor<?x4x2x3x5xf32>) -> ()
  }) : () -> ()
}) : () -> ()

