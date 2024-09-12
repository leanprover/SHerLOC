"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x5x3xf32>) -> tensor<?x1x0xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x5x3xf32>):
    %0 = "stablehlo.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %1 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %2 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %3 = "stablehlo.concatenate"(%0, %1, %2) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %4 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %5 = "stablehlo.reshape"(%4) : (tensor<i32>) -> tensor<1xi32>
    %6 = "stablehlo.constant"() <{value = dense<2> : tensor<1xi32>}> : () -> tensor<1xi32>
    %7 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %8 = "stablehlo.concatenate"(%5, %6, %7) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %9 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %10 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %11 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %12 = "stablehlo.concatenate"(%9, %10, %11) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %13 = "stablehlo.real_dynamic_slice"(%arg1, %3, %8, %12) : (tensor<?x5x3xf32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<?x1x0xf32>
    "func.return"(%13) : (tensor<?x1x0xf32>) -> ()
  }) : () -> ()
}) : () -> ()

