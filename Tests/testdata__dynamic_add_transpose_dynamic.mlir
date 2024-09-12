"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x4xf32>) -> tensor<?x4xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x4xf32>):
    %0 = "stablehlo.cosine"(%arg1) : (tensor<?x4xf32>) -> tensor<?x4xf32>
    %1 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %2 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %3 = "stablehlo.reshape"(%2) : (tensor<i32>) -> tensor<1xi32>
    %4 = "stablehlo.constant"() <{value = dense<4> : tensor<1xi32>}> : () -> tensor<1xi32>
    %5 = "stablehlo.concatenate"(%3, %4) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %6 = "stablehlo.dynamic_broadcast_in_dim"(%1, %5) <{broadcast_dimensions = array<i64>}> : (tensor<f32>, tensor<2xi32>) -> tensor<?x4xf32>
    %7 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %8 = "stablehlo.reduce"(%6, %7) <{dimensions = array<i64: 0>}> ({
    ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>):
      %20 = "stablehlo.add"(%arg4, %arg5) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%20) : (tensor<f32>) -> ()
    }) : (tensor<?x4xf32>, tensor<f32>) -> tensor<4xf32>
    %9 = "stablehlo.reshape"(%8) : (tensor<4xf32>) -> tensor<1x4xf32>
    %10 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %11 = "stablehlo.reduce"(%9, %10) <{dimensions = array<i64: 0>}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %19 = "stablehlo.add"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%19) : (tensor<f32>) -> ()
    }) : (tensor<1x4xf32>, tensor<f32>) -> tensor<4xf32>
    %12 = "stablehlo.multiply"(%6, %0) : (tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<?x4xf32>
    %13 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %14 = "stablehlo.reshape"(%13) : (tensor<i32>) -> tensor<1xi32>
    %15 = "stablehlo.constant"() <{value = dense<4> : tensor<1xi32>}> : () -> tensor<1xi32>
    %16 = "stablehlo.concatenate"(%14, %15) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %17 = "stablehlo.dynamic_broadcast_in_dim"(%11, %16) <{broadcast_dimensions = array<i64: 1>}> : (tensor<4xf32>, tensor<2xi32>) -> tensor<?x4xf32>
    %18 = "stablehlo.add"(%12, %17) : (tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<?x4xf32>
    "func.return"(%18) : (tensor<?x4xf32>) -> ()
  }) : () -> ()
}) : () -> ()

