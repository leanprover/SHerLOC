"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x5x6x7xf32>, tensor<?x5x2x2xf32>) -> tensor<?x5x6x7xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x5x6x7xf32>, %arg2: tensor<?x5x2x2xf32>):
    %0 = "stablehlo.constant"() <{value = dense<[[[0, 1], [2, 3]], [[4, 0], [1, 2]]]> : tensor<2x2x2xi64>}> : () -> tensor<2x2x2xi64>
    %1 = "stablehlo.scatter"(%arg1, %0, %arg2) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [2, 3], scatter_dims_to_operand_dims = [2, 3], index_vector_dim = 2>, unique_indices = true}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      "stablehlo.return"(%arg4) : (tensor<f32>) -> ()
    }) : (tensor<?x5x6x7xf32>, tensor<2x2x2xi64>, tensor<?x5x2x2xf32>) -> tensor<?x5x6x7xf32>
    "func.return"(%1) : (tensor<?x5x6x7xf32>) -> ()
  }) : () -> ()
}) : () -> ()

