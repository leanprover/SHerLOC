"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x5xf32>, tensor<?x2xf32>) -> tensor<?x5xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x5xf32>, %arg2: tensor<?x2xf32>):
    %0 = "stablehlo.constant"() <{value = dense<[[0], [2]]> : tensor<2x1xi64>}> : () -> tensor<2x1xi64>
    %1 = "stablehlo.scatter"(%arg1, %0, %arg2) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      "stablehlo.return"(%arg4) : (tensor<f32>) -> ()
    }) : (tensor<?x5xf32>, tensor<2x1xi64>, tensor<?x2xf32>) -> tensor<?x5xf32>
    "func.return"(%1) : (tensor<?x5xf32>) -> ()
  }) : () -> ()
}) : () -> ()

