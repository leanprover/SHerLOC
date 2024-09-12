"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}, {mhlo.sharding = ""}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x4xf32>, tensor<2x1xi32>, tensor<?x2xf32>) -> tensor<?x4xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x4xf32>, %arg2: tensor<2x1xi32>, %arg3: tensor<?x2xf32>):
    %0 = "stablehlo.scatter"(%arg1, %arg2, %arg3) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>):
      %1 = "stablehlo.add"(%arg4, %arg5) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%1) : (tensor<f32>) -> ()
    }) : (tensor<?x4xf32>, tensor<2x1xi32>, tensor<?x2xf32>) -> tensor<?x4xf32>
    "func.return"(%0) : (tensor<?x4xf32>) -> ()
  }) : () -> ()
}) : () -> ()

