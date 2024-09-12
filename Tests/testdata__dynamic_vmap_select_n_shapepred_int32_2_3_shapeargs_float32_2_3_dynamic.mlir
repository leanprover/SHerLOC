"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}, {mhlo.sharding = ""}, {mhlo.sharding = ""}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x2x3xi32>, tensor<?x2x3xf32>, tensor<?x2x3xf32>, tensor<?x2x3xf32>) -> tensor<?x2x3xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x2x3xi32>, %arg2: tensor<?x2x3xf32>, %arg3: tensor<?x2x3xf32>, %arg4: tensor<?x2x3xf32>):
    %0 = "stablehlo.constant"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
    %1 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %2 = "stablehlo.reshape"(%1) : (tensor<i32>) -> tensor<1xi32>
    %3 = "stablehlo.constant"() <{value = dense<2> : tensor<1xi32>}> : () -> tensor<1xi32>
    %4 = "stablehlo.constant"() <{value = dense<3> : tensor<1xi32>}> : () -> tensor<1xi32>
    %5 = "stablehlo.concatenate"(%2, %3, %4) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %6 = "stablehlo.dynamic_broadcast_in_dim"(%0, %5) <{broadcast_dimensions = array<i64>}> : (tensor<i32>, tensor<3xi32>) -> tensor<?x2x3xi32>
    %7 = "stablehlo.compare"(%arg1, %6) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<?x2x3xi32>, tensor<?x2x3xi32>) -> tensor<?x2x3xi1>
    %8 = "stablehlo.constant"() <{value = dense<2> : tensor<i32>}> : () -> tensor<i32>
    %9 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %10 = "stablehlo.reshape"(%9) : (tensor<i32>) -> tensor<1xi32>
    %11 = "stablehlo.constant"() <{value = dense<2> : tensor<1xi32>}> : () -> tensor<1xi32>
    %12 = "stablehlo.constant"() <{value = dense<3> : tensor<1xi32>}> : () -> tensor<1xi32>
    %13 = "stablehlo.concatenate"(%10, %11, %12) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %14 = "stablehlo.dynamic_broadcast_in_dim"(%8, %13) <{broadcast_dimensions = array<i64>}> : (tensor<i32>, tensor<3xi32>) -> tensor<?x2x3xi32>
    %15 = "stablehlo.compare"(%arg1, %14) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<?x2x3xi32>, tensor<?x2x3xi32>) -> tensor<?x2x3xi1>
    %16 = "stablehlo.select"(%15, %arg3, %arg4) : (tensor<?x2x3xi1>, tensor<?x2x3xf32>, tensor<?x2x3xf32>) -> tensor<?x2x3xf32>
    %17 = "stablehlo.select"(%7, %arg2, %16) : (tensor<?x2x3xi1>, tensor<?x2x3xf32>, tensor<?x2x3xf32>) -> tensor<?x2x3xf32>
    "func.return"(%17) : (tensor<?x2x3xf32>) -> ()
  }) : () -> ()
}) : () -> ()

