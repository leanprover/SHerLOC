"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x2xf32>) -> tensor<?x?xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x2xf32>):
    %0 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %1 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %2 = "stablehlo.reshape"(%1) : (tensor<i32>) -> tensor<1xi32>
    %3 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %4 = "stablehlo.constant"() <{value = dense<2> : tensor<1xi32>}> : () -> tensor<1xi32>
    %5 = "stablehlo.concatenate"(%0, %2, %3, %4) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
    %6 = "stablehlo.dynamic_reshape"(%arg1, %5) : (tensor<?x2xf32>, tensor<4xi32>) -> tensor<1x?x1x2xf32>
    %7 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %8 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %9 = "stablehlo.reshape"(%8) : (tensor<i32>) -> tensor<1xi32>
    %10 = "stablehlo.constant"() <{value = dense<2> : tensor<1xi32>}> : () -> tensor<1xi32>
    %11 = "stablehlo.concatenate"(%7, %9, %10) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %12 = "stablehlo.dynamic_reshape"(%6, %11) : (tensor<1x?x1x2xf32>, tensor<3xi32>) -> tensor<1x?x2xf32>
    %13 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %14 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %15 = "stablehlo.reshape"(%14) : (tensor<i32>) -> tensor<1xi32>
    %16 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %17 = "stablehlo.reshape"(%16) : (tensor<i32>) -> tensor<1xi32>
    %18 = "stablehlo.constant"() <{value = dense<2> : tensor<1xi32>}> : () -> tensor<1xi32>
    %19 = "stablehlo.concatenate"(%13, %15, %17, %18) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
    %20 = "stablehlo.dynamic_broadcast_in_dim"(%12, %19) <{broadcast_dimensions = array<i64: 0, 1, 3>}> : (tensor<1x?x2xf32>, tensor<4xi32>) -> tensor<1x?x?x2xf32>
    %21 = "stablehlo.constant"() <{value = dense<2> : tensor<i64>}> : () -> tensor<i64>
    %22 = "stablehlo.multiply"(%arg0, %21) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %23 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %24 = "stablehlo.reshape"(%23) : (tensor<i32>) -> tensor<1xi32>
    %25 = "stablehlo.convert"(%22) : (tensor<i64>) -> tensor<i32>
    %26 = "stablehlo.reshape"(%25) : (tensor<i32>) -> tensor<1xi32>
    %27 = "stablehlo.concatenate"(%24, %26) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %28 = "stablehlo.dynamic_reshape"(%20, %27) : (tensor<1x?x?x2xf32>, tensor<2xi32>) -> tensor<?x?xf32>
    "func.return"(%28) : (tensor<?x?xf32>) -> ()
  }) : () -> ()
}) : () -> ()

