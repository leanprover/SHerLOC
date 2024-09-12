"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x8x4xf32>, tensor<?x8x4xi1>) -> tensor<f32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg4: tensor<i64>, %arg5: tensor<?x8x4xf32>, %arg6: tensor<?x8x4xi1>):
    %7 = "stablehlo.convert"(%arg6) : (tensor<?x8x4xi1>) -> tensor<?x8x4xi32>
    %8 = "stablehlo.convert"(%7) : (tensor<?x8x4xi32>) -> tensor<?x8x4xi64>
    %9 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %10 = "stablehlo.reduce"(%8, %9) <{dimensions = array<i64: 0, 1, 2>}> ({
    ^bb0(%arg9: tensor<i64>, %arg10: tensor<i64>):
      %18 = "stablehlo.add"(%arg9, %arg10) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      "stablehlo.return"(%18) : (tensor<i64>) -> ()
    }) : (tensor<?x8x4xi64>, tensor<i64>) -> tensor<i64>
    %11 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %12 = "func.call"(%arg4, %arg6, %arg5, %11) <{callee = @_where}> : (tensor<i64>, tensor<?x8x4xi1>, tensor<?x8x4xf32>, tensor<f32>) -> tensor<?x8x4xf32>
    %13 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %14 = "stablehlo.reduce"(%12, %13) <{dimensions = array<i64: 0, 1, 2>}> ({
    ^bb0(%arg7: tensor<f32>, %arg8: tensor<f32>):
      %17 = "stablehlo.add"(%arg7, %arg8) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%17) : (tensor<f32>) -> ()
    }) : (tensor<?x8x4xf32>, tensor<f32>) -> tensor<f32>
    %15 = "stablehlo.convert"(%10) : (tensor<i64>) -> tensor<f32>
    %16 = "stablehlo.divide"(%14, %15) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "func.return"(%16) : (tensor<f32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<i64>, tensor<?x8x4xi1>, tensor<?x8x4xf32>, tensor<f32>) -> tensor<?x8x4xf32>, sym_name = "_where", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x8x4xi1>, %arg2: tensor<?x8x4xf32>, %arg3: tensor<f32>):
    %0 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %1 = "stablehlo.reshape"(%0) : (tensor<i32>) -> tensor<1xi32>
    %2 = "stablehlo.constant"() <{value = dense<8> : tensor<1xi32>}> : () -> tensor<1xi32>
    %3 = "stablehlo.constant"() <{value = dense<4> : tensor<1xi32>}> : () -> tensor<1xi32>
    %4 = "stablehlo.concatenate"(%1, %2, %3) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %5 = "stablehlo.dynamic_broadcast_in_dim"(%arg3, %4) <{broadcast_dimensions = array<i64>}> : (tensor<f32>, tensor<3xi32>) -> tensor<?x8x4xf32>
    %6 = "stablehlo.select"(%arg1, %arg2, %5) : (tensor<?x8x4xi1>, tensor<?x8x4xf32>, tensor<?x8x4xf32>) -> tensor<?x8x4xf32>
    "func.return"(%6) : (tensor<?x8x4xf32>) -> ()
  }) : () -> ()
}) : () -> ()

