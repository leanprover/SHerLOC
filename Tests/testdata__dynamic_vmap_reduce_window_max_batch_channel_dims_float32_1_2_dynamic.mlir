"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x1x2xf32>) -> tensor<?x1x1xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x1x2xf32>):
    %0 = "stablehlo.constant"() <{value = dense<0xFF800000> : tensor<f32>}> : () -> tensor<f32>
    %1 = "stablehlo.broadcast_in_dim"(%0) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<f32>
    %2 = "stablehlo.reduce_window"(%arg1, %1) <{window_dimensions = array<i64: 1, 1, 2>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %3 = "stablehlo.maximum"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%3) : (tensor<f32>) -> ()
    }) : (tensor<?x1x2xf32>, tensor<f32>) -> tensor<?x1x1xf32>
    "func.return"(%2) : (tensor<?x1x1xf32>) -> ()
  }) : () -> ()
}) : () -> ()

