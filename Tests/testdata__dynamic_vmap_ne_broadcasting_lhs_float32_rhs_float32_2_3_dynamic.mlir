"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?xf32>, tensor<?x2x3xf32>) -> tensor<?x2x3xi1>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?xf32>, %arg2: tensor<?x2x3xf32>):
    %0 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %1 = "stablehlo.reshape"(%0) : (tensor<i32>) -> tensor<1xi32>
    %2 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %3 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %4 = "stablehlo.concatenate"(%1, %2, %3) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %5 = "stablehlo.dynamic_broadcast_in_dim"(%arg1, %4) <{broadcast_dimensions = array<i64: 0>}> : (tensor<?xf32>, tensor<3xi32>) -> tensor<?x1x1xf32>
    %6 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %7 = "stablehlo.reshape"(%6) : (tensor<i32>) -> tensor<1xi32>
    %8 = "stablehlo.constant"() <{value = dense<2> : tensor<1xi32>}> : () -> tensor<1xi32>
    %9 = "stablehlo.constant"() <{value = dense<3> : tensor<1xi32>}> : () -> tensor<1xi32>
    %10 = "stablehlo.concatenate"(%7, %8, %9) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %11 = "stablehlo.dynamic_broadcast_in_dim"(%5, %10) <{broadcast_dimensions = array<i64: 0, 1, 2>}> : (tensor<?x1x1xf32>, tensor<3xi32>) -> tensor<?x2x3xf32>
    %12 = "stablehlo.compare"(%11, %arg2) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction NE>}> : (tensor<?x2x3xf32>, tensor<?x2x3xf32>) -> tensor<?x2x3xi1>
    "func.return"(%12) : (tensor<?x2x3xi1>) -> ()
  }) : () -> ()
}) : () -> ()

