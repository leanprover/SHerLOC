"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x8x4xf32>, tensor<?x8x4xf32>) -> tensor<f32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x8x4xf32>, %arg2: tensor<?x8x4xf32>):
    %0 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %1 = "stablehlo.reduce"(%arg2, %0) <{dimensions = array<i64: 0, 1, 2>}> ({
    ^bb0(%arg5: tensor<f32>, %arg6: tensor<f32>):
      %7 = "stablehlo.add"(%arg5, %arg6) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%7) : (tensor<f32>) -> ()
    }) : (tensor<?x8x4xf32>, tensor<f32>) -> tensor<f32>
    %2 = "stablehlo.multiply"(%arg1, %arg2) : (tensor<?x8x4xf32>, tensor<?x8x4xf32>) -> tensor<?x8x4xf32>
    %3 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %4 = "stablehlo.reduce"(%2, %3) <{dimensions = array<i64: 0, 1, 2>}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %6 = "stablehlo.add"(%arg3, %arg4) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%6) : (tensor<f32>) -> ()
    }) : (tensor<?x8x4xf32>, tensor<f32>) -> tensor<f32>
    %5 = "stablehlo.divide"(%4, %1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "func.return"(%5) : (tensor<f32>) -> ()
  }) : () -> ()
}) : () -> ()

