"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x4xf32>) -> tensor<?x4xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x4xf32>):
    %0 = "stablehlo.constant"() <{value = dense<-1> : tensor<i64>}> : () -> tensor<i64>
    %1 = "stablehlo.add"(%0, %arg0) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %2 = "stablehlo.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %3 = "stablehlo.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %4 = "stablehlo.concatenate"(%2, %3) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %5 = "stablehlo.convert"(%1) : (tensor<i64>) -> tensor<i32>
    %6 = "stablehlo.reshape"(%5) : (tensor<i32>) -> tensor<1xi32>
    %7 = "stablehlo.constant"() <{value = dense<4> : tensor<1xi32>}> : () -> tensor<1xi32>
    %8 = "stablehlo.concatenate"(%6, %7) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %9 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %10 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %11 = "stablehlo.concatenate"(%9, %10) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %12 = "stablehlo.real_dynamic_slice"(%arg1, %4, %8, %11) : (tensor<?x4xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x4xf32>
    "func.return"(%12) : (tensor<?x4xf32>) -> ()
  }) : () -> ()
}) : () -> ()

