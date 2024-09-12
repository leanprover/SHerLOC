"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x3xf32>, tensor<3xf32>) -> tensor<?x3xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x3xf32>, %arg2: tensor<3xf32>):
    %0 = "stablehlo.constant"() <{value = dense<5.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %1 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %2 = "stablehlo.reshape"(%1) : (tensor<i32>) -> tensor<1xi32>
    %3 = "stablehlo.constant"() <{value = dense<3> : tensor<1xi32>}> : () -> tensor<1xi32>
    %4 = "stablehlo.concatenate"(%2, %3) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %5 = "stablehlo.dynamic_broadcast_in_dim"(%0, %4) <{broadcast_dimensions = array<i64>}> : (tensor<f32>, tensor<2xi32>) -> tensor<?x3xf32>
    %6 = "stablehlo.compare"(%arg1, %5) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<?x3xf32>, tensor<?x3xf32>) -> tensor<?x3xi1>
    %7 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %8 = "stablehlo.reshape"(%7) : (tensor<i32>) -> tensor<1xi32>
    %9 = "stablehlo.constant"() <{value = dense<3> : tensor<1xi32>}> : () -> tensor<1xi32>
    %10 = "stablehlo.concatenate"(%8, %9) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %11 = "stablehlo.dynamic_broadcast_in_dim"(%arg2, %10) <{broadcast_dimensions = array<i64: 1>}> : (tensor<3xf32>, tensor<2xi32>) -> tensor<?x3xf32>
    %12 = "stablehlo.select"(%6, %arg1, %11) : (tensor<?x3xi1>, tensor<?x3xf32>, tensor<?x3xf32>) -> tensor<?x3xf32>
    "func.return"(%12) : (tensor<?x3xf32>) -> ()
  }) : () -> ()
}) : () -> ()

