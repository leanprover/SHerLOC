"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x4xf32>) -> tensor<?x4xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x4xf32>):
    %0 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %1 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %2 = "stablehlo.convert"(%0) : (tensor<i64>) -> tensor<i32>
    %3 = "stablehlo.reshape"(%2) : (tensor<i32>) -> tensor<1xi32>
    %4 = "stablehlo.convert"(%1) : (tensor<i64>) -> tensor<i32>
    %5 = "stablehlo.reshape"(%4) : (tensor<i32>) -> tensor<1xi32>
    %6 = "stablehlo.concatenate"(%3, %5) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %7 = "stablehlo.convert"(%0) : (tensor<i64>) -> tensor<i32>
    %8 = "stablehlo.reshape"(%7) : (tensor<i32>) -> tensor<1xi32>
    %9 = "stablehlo.convert"(%1) : (tensor<i64>) -> tensor<i32>
    %10 = "stablehlo.reshape"(%9) : (tensor<i32>) -> tensor<1xi32>
    %11 = "stablehlo.concatenate"(%8, %10) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %12 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %13 = "stablehlo.reshape"(%12) : (tensor<i32>) -> tensor<1xi32>
    %14 = "stablehlo.constant"() <{value = dense<4> : tensor<1xi32>}> : () -> tensor<1xi32>
    %15 = "stablehlo.concatenate"(%13, %14) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %16 = "stablehlo.add"(%11, %15) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
    %17 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %18 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %19 = "stablehlo.concatenate"(%17, %18) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %20 = "stablehlo.real_dynamic_slice"(%arg1, %6, %16, %19) : (tensor<?x4xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x4xf32>
    "func.return"(%20) : (tensor<?x4xf32>) -> ()
  }) : () -> ()
}) : () -> ()

