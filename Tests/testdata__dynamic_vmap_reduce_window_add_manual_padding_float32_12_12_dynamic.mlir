"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x12x12xf32>) -> tensor<?x6x6xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x12x12xf32>):
    %0 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %1 = "stablehlo.broadcast_in_dim"(%0) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<f32>
    %2 = "stablehlo.reduce_window"(%arg1, %1) <{padding = dense<[[0, 0], [0, 1], [0, 1]]> : tensor<3x2xi64>, window_dimensions = array<i64: 1, 3, 3>, window_strides = array<i64: 1, 2, 2>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %3 = "stablehlo.add"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%3) : (tensor<f32>) -> ()
    }) : (tensor<?x12x12xf32>, tensor<f32>) -> tensor<?x6x6xf32>
    "func.return"(%2) : (tensor<?x6x6xf32>) -> ()
  }) : () -> ()
}) : () -> ()

