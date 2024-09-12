"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x4xf32>) -> tensor<?x4xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x4xf32>):
    %0 = "stablehlo.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %1 = "stablehlo.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %2 = "stablehlo.concatenate"(%0, %1) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %3 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %4 = "stablehlo.reshape"(%3) : (tensor<i32>) -> tensor<1xi32>
    %5 = "stablehlo.constant"() <{value = dense<4> : tensor<1xi32>}> : () -> tensor<1xi32>
    %6 = "stablehlo.concatenate"(%4, %5) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %7 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %8 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %9 = "stablehlo.concatenate"(%7, %8) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %10 = "stablehlo.real_dynamic_slice"(%arg1, %2, %6, %9) : (tensor<?x4xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x4xf32>
    "func.return"(%10) : (tensor<?x4xf32>) -> ()
  }) : () -> ()
}) : () -> ()

