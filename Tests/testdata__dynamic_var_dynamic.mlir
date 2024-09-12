"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x8x4xf32>, tensor<?x8x4xi1>) -> tensor<f32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg20: tensor<i64>, %arg21: tensor<?x8x4xf32>, %arg22: tensor<?x8x4xi1>):
    %48 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %49 = "func.call"(%arg20, %arg21, %48, %arg22) <{callee = @_var}> : (tensor<i64>, tensor<?x8x4xf32>, tensor<i64>, tensor<?x8x4xi1>) -> tensor<f32>
    "func.return"(%49) : (tensor<f32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<i64>, tensor<?x8x4xf32>, tensor<i64>, tensor<?x8x4xi1>) -> tensor<f32>, sym_name = "_var", sym_visibility = "private"}> ({
  ^bb0(%arg8: tensor<i64>, %arg9: tensor<?x8x4xf32>, %arg10: tensor<i64>, %arg11: tensor<?x8x4xi1>):
    %14 = "stablehlo.convert"(%arg11) : (tensor<?x8x4xi1>) -> tensor<?x8x4xi32>
    %15 = "stablehlo.convert"(%14) : (tensor<?x8x4xi32>) -> tensor<?x8x4xf32>
    %16 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %17 = "stablehlo.reduce"(%15, %16) <{dimensions = array<i64: 0, 1, 2>}> ({
    ^bb0(%arg18: tensor<f32>, %arg19: tensor<f32>):
      %47 = "stablehlo.add"(%arg18, %arg19) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%47) : (tensor<f32>) -> ()
    }) : (tensor<?x8x4xf32>, tensor<f32>) -> tensor<f32>
    %18 = "stablehlo.broadcast_in_dim"(%17) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<1x1x1xf32>
    %19 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %20 = "func.call"(%arg8, %arg11, %arg9, %19) <{callee = @_where}> : (tensor<i64>, tensor<?x8x4xi1>, tensor<?x8x4xf32>, tensor<f32>) -> tensor<?x8x4xf32>
    %21 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %22 = "stablehlo.reduce"(%20, %21) <{dimensions = array<i64: 0, 1, 2>}> ({
    ^bb0(%arg16: tensor<f32>, %arg17: tensor<f32>):
      %46 = "stablehlo.add"(%arg16, %arg17) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%46) : (tensor<f32>) -> ()
    }) : (tensor<?x8x4xf32>, tensor<f32>) -> tensor<f32>
    %23 = "stablehlo.broadcast_in_dim"(%22) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<1x1x1xf32>
    %24 = "stablehlo.divide"(%23, %18) : (tensor<1x1x1xf32>, tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
    %25 = "stablehlo.convert"(%arg8) : (tensor<i64>) -> tensor<i32>
    %26 = "stablehlo.reshape"(%25) : (tensor<i32>) -> tensor<1xi32>
    %27 = "stablehlo.constant"() <{value = dense<8> : tensor<1xi32>}> : () -> tensor<1xi32>
    %28 = "stablehlo.constant"() <{value = dense<4> : tensor<1xi32>}> : () -> tensor<1xi32>
    %29 = "stablehlo.concatenate"(%26, %27, %28) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %30 = "stablehlo.dynamic_broadcast_in_dim"(%24, %29) <{broadcast_dimensions = array<i64: 0, 1, 2>}> : (tensor<1x1x1xf32>, tensor<3xi32>) -> tensor<?x8x4xf32>
    %31 = "stablehlo.subtract"(%arg9, %30) : (tensor<?x8x4xf32>, tensor<?x8x4xf32>) -> tensor<?x8x4xf32>
    %32 = "stablehlo.multiply"(%31, %31) : (tensor<?x8x4xf32>, tensor<?x8x4xf32>) -> tensor<?x8x4xf32>
    %33 = "stablehlo.convert"(%arg11) : (tensor<?x8x4xi1>) -> tensor<?x8x4xi32>
    %34 = "stablehlo.convert"(%33) : (tensor<?x8x4xi32>) -> tensor<?x8x4xf32>
    %35 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %36 = "stablehlo.reduce"(%34, %35) <{dimensions = array<i64: 0, 1, 2>}> ({
    ^bb0(%arg14: tensor<f32>, %arg15: tensor<f32>):
      %45 = "stablehlo.add"(%arg14, %arg15) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%45) : (tensor<f32>) -> ()
    }) : (tensor<?x8x4xf32>, tensor<f32>) -> tensor<f32>
    %37 = "stablehlo.convert"(%arg10) : (tensor<i64>) -> tensor<f32>
    %38 = "stablehlo.subtract"(%36, %37) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %39 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %40 = "func.call"(%arg8, %arg11, %32, %39) <{callee = @_where_0}> : (tensor<i64>, tensor<?x8x4xi1>, tensor<?x8x4xf32>, tensor<f32>) -> tensor<?x8x4xf32>
    %41 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %42 = "stablehlo.reduce"(%40, %41) <{dimensions = array<i64: 0, 1, 2>}> ({
    ^bb0(%arg12: tensor<f32>, %arg13: tensor<f32>):
      %44 = "stablehlo.add"(%arg12, %arg13) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%44) : (tensor<f32>) -> ()
    }) : (tensor<?x8x4xf32>, tensor<f32>) -> tensor<f32>
    %43 = "stablehlo.divide"(%42, %38) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "func.return"(%43) : (tensor<f32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<i64>, tensor<?x8x4xi1>, tensor<?x8x4xf32>, tensor<f32>) -> tensor<?x8x4xf32>, sym_name = "_where", sym_visibility = "private"}> ({
  ^bb0(%arg4: tensor<i64>, %arg5: tensor<?x8x4xi1>, %arg6: tensor<?x8x4xf32>, %arg7: tensor<f32>):
    %7 = "stablehlo.convert"(%arg4) : (tensor<i64>) -> tensor<i32>
    %8 = "stablehlo.reshape"(%7) : (tensor<i32>) -> tensor<1xi32>
    %9 = "stablehlo.constant"() <{value = dense<8> : tensor<1xi32>}> : () -> tensor<1xi32>
    %10 = "stablehlo.constant"() <{value = dense<4> : tensor<1xi32>}> : () -> tensor<1xi32>
    %11 = "stablehlo.concatenate"(%8, %9, %10) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %12 = "stablehlo.dynamic_broadcast_in_dim"(%arg7, %11) <{broadcast_dimensions = array<i64>}> : (tensor<f32>, tensor<3xi32>) -> tensor<?x8x4xf32>
    %13 = "stablehlo.select"(%arg5, %arg6, %12) : (tensor<?x8x4xi1>, tensor<?x8x4xf32>, tensor<?x8x4xf32>) -> tensor<?x8x4xf32>
    "func.return"(%13) : (tensor<?x8x4xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<i64>, tensor<?x8x4xi1>, tensor<?x8x4xf32>, tensor<f32>) -> tensor<?x8x4xf32>, sym_name = "_where_0", sym_visibility = "private"}> ({
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

