"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x1xf32>) -> tensor<?x2xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x1xf32>):
    %0 = "stablehlo.constant"() <{value = dense<3.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %1 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %2 = "stablehlo.reshape"(%1) : (tensor<i32>) -> tensor<1xi32>
    %3 = "stablehlo.constant"() <{value = dense<2> : tensor<1xi32>}> : () -> tensor<1xi32>
    %4 = "stablehlo.concatenate"(%2, %3) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %5 = "stablehlo.dynamic_broadcast_in_dim"(%0, %4) <{broadcast_dimensions = array<i64>}> : (tensor<f64>, tensor<2xi32>) -> tensor<?x2xf64>
    %6 = "stablehlo.convert"(%5) : (tensor<?x2xf64>) -> tensor<?x2xf32>
    %7 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %8 = "stablehlo.reshape"(%7) : (tensor<i32>) -> tensor<1xi32>
    %9 = "stablehlo.constant"() <{value = dense<2> : tensor<1xi32>}> : () -> tensor<1xi32>
    %10 = "stablehlo.concatenate"(%8, %9) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %11 = "stablehlo.dynamic_broadcast_in_dim"(%arg1, %10) <{broadcast_dimensions = array<i64: 0, 1>}> : (tensor<?x1xf32>, tensor<2xi32>) -> tensor<?x2xf32>
    %12 = "stablehlo.add"(%6, %11) : (tensor<?x2xf32>, tensor<?x2xf32>) -> tensor<?x2xf32>
    "func.return"(%12) : (tensor<?x2xf32>) -> ()
  }) : () -> ()
}) : () -> ()

