"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x3xf32>) -> tensor<?x6xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x3xf32>):
    %0 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %1 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %2 = "stablehlo.reshape"(%1) : (tensor<i32>) -> tensor<1xi32>
    %3 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %4 = "stablehlo.constant"() <{value = dense<3> : tensor<1xi32>}> : () -> tensor<1xi32>
    %5 = "stablehlo.concatenate"(%0, %2, %3, %4) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
    %6 = "stablehlo.dynamic_reshape"(%arg1, %5) : (tensor<?x3xf32>, tensor<4xi32>) -> tensor<1x?x1x3xf32>
    %7 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %8 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %9 = "stablehlo.reshape"(%8) : (tensor<i32>) -> tensor<1xi32>
    %10 = "stablehlo.constant"() <{value = dense<3> : tensor<1xi32>}> : () -> tensor<1xi32>
    %11 = "stablehlo.concatenate"(%7, %9, %10) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %12 = "stablehlo.dynamic_reshape"(%6, %11) : (tensor<1x?x1x3xf32>, tensor<3xi32>) -> tensor<1x?x3xf32>
    %13 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %14 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %15 = "stablehlo.reshape"(%14) : (tensor<i32>) -> tensor<1xi32>
    %16 = "stablehlo.constant"() <{value = dense<2> : tensor<1xi32>}> : () -> tensor<1xi32>
    %17 = "stablehlo.constant"() <{value = dense<3> : tensor<1xi32>}> : () -> tensor<1xi32>
    %18 = "stablehlo.concatenate"(%13, %15, %16, %17) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
    %19 = "stablehlo.dynamic_broadcast_in_dim"(%12, %18) <{broadcast_dimensions = array<i64: 0, 1, 3>}> : (tensor<1x?x3xf32>, tensor<4xi32>) -> tensor<1x?x2x3xf32>
    %20 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %21 = "stablehlo.reshape"(%20) : (tensor<i32>) -> tensor<1xi32>
    %22 = "stablehlo.constant"() <{value = dense<6> : tensor<1xi32>}> : () -> tensor<1xi32>
    %23 = "stablehlo.concatenate"(%21, %22) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %24 = "stablehlo.dynamic_reshape"(%19, %23) : (tensor<1x?x2x3xf32>, tensor<2xi32>) -> tensor<?x6xf32>
    "func.return"(%24) : (tensor<?x6xf32>) -> ()
  }) : () -> ()
}) : () -> ()

