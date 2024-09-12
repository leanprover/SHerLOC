"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<i64>, tensor<?x1x?xf32>) -> tensor<?x1x?xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>, %arg2: tensor<?x1x?xf32>):
    %0 = "stablehlo.cosine"(%arg2) : (tensor<?x1x?xf32>) -> tensor<?x1x?xf32>
    %1 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %2 = "stablehlo.constant"() <{value = dense<2> : tensor<1xi32>}> : () -> tensor<1xi32>
    %3 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %4 = "stablehlo.reshape"(%3) : (tensor<i32>) -> tensor<1xi32>
    %5 = "stablehlo.constant"() <{value = dense<5> : tensor<1xi32>}> : () -> tensor<1xi32>
    %6 = "stablehlo.convert"(%arg1) : (tensor<i64>) -> tensor<i32>
    %7 = "stablehlo.reshape"(%6) : (tensor<i32>) -> tensor<1xi32>
    %8 = "stablehlo.constant"() <{value = dense<4> : tensor<1xi32>}> : () -> tensor<1xi32>
    %9 = "stablehlo.concatenate"(%2, %4, %5, %7, %8) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<5xi32>
    %10 = "stablehlo.dynamic_broadcast_in_dim"(%1, %9) <{broadcast_dimensions = array<i64>}> : (tensor<f32>, tensor<5xi32>) -> tensor<2x?x5x?x4xf32>
    %11 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %12 = "stablehlo.reduce"(%10, %11) <{dimensions = array<i64: 0, 2, 4>}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %21 = "stablehlo.add"(%arg3, %arg4) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%21) : (tensor<f32>) -> ()
    }) : (tensor<2x?x5x?x4xf32>, tensor<f32>) -> tensor<?x?xf32>
    %13 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %14 = "stablehlo.reshape"(%13) : (tensor<i32>) -> tensor<1xi32>
    %15 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %16 = "stablehlo.convert"(%arg1) : (tensor<i64>) -> tensor<i32>
    %17 = "stablehlo.reshape"(%16) : (tensor<i32>) -> tensor<1xi32>
    %18 = "stablehlo.concatenate"(%14, %15, %17) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %19 = "stablehlo.dynamic_broadcast_in_dim"(%12, %18) <{broadcast_dimensions = array<i64: 0, 2>}> : (tensor<?x?xf32>, tensor<3xi32>) -> tensor<?x1x?xf32>
    %20 = "stablehlo.multiply"(%19, %0) : (tensor<?x1x?xf32>, tensor<?x1x?xf32>) -> tensor<?x1x?xf32>
    "func.return"(%20) : (tensor<?x1x?xf32>) -> ()
  }) : () -> ()
}) : () -> ()

