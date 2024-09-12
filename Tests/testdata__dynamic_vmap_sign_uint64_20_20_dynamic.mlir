"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x20x20xui64>) -> tensor<?x20x20xui64>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x20x20xui64>):
    %0 = "stablehlo.constant"() <{value = dense<0> : tensor<ui64>}> : () -> tensor<ui64>
    %1 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %2 = "stablehlo.reshape"(%1) : (tensor<i32>) -> tensor<1xi32>
    %3 = "stablehlo.constant"() <{value = dense<20> : tensor<1xi32>}> : () -> tensor<1xi32>
    %4 = "stablehlo.constant"() <{value = dense<20> : tensor<1xi32>}> : () -> tensor<1xi32>
    %5 = "stablehlo.concatenate"(%2, %3, %4) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %6 = "stablehlo.dynamic_broadcast_in_dim"(%0, %5) <{broadcast_dimensions = array<i64>}> : (tensor<ui64>, tensor<3xi32>) -> tensor<?x20x20xui64>
    %7 = "stablehlo.compare"(%arg1, %6) <{compare_type = #stablehlo<comparison_type UNSIGNED>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<?x20x20xui64>, tensor<?x20x20xui64>) -> tensor<?x20x20xi1>
    %8 = "stablehlo.constant"() <{value = dense<0> : tensor<ui64>}> : () -> tensor<ui64>
    %9 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %10 = "stablehlo.reshape"(%9) : (tensor<i32>) -> tensor<1xi32>
    %11 = "stablehlo.constant"() <{value = dense<20> : tensor<1xi32>}> : () -> tensor<1xi32>
    %12 = "stablehlo.constant"() <{value = dense<20> : tensor<1xi32>}> : () -> tensor<1xi32>
    %13 = "stablehlo.concatenate"(%10, %11, %12) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %14 = "stablehlo.dynamic_broadcast_in_dim"(%8, %13) <{broadcast_dimensions = array<i64>}> : (tensor<ui64>, tensor<3xi32>) -> tensor<?x20x20xui64>
    %15 = "stablehlo.constant"() <{value = dense<1> : tensor<ui64>}> : () -> tensor<ui64>
    %16 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %17 = "stablehlo.reshape"(%16) : (tensor<i32>) -> tensor<1xi32>
    %18 = "stablehlo.constant"() <{value = dense<20> : tensor<1xi32>}> : () -> tensor<1xi32>
    %19 = "stablehlo.constant"() <{value = dense<20> : tensor<1xi32>}> : () -> tensor<1xi32>
    %20 = "stablehlo.concatenate"(%17, %18, %19) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %21 = "stablehlo.dynamic_broadcast_in_dim"(%15, %20) <{broadcast_dimensions = array<i64>}> : (tensor<ui64>, tensor<3xi32>) -> tensor<?x20x20xui64>
    %22 = "stablehlo.select"(%7, %14, %21) : (tensor<?x20x20xi1>, tensor<?x20x20xui64>, tensor<?x20x20xui64>) -> tensor<?x20x20xui64>
    "func.return"(%22) : (tensor<?x20x20xui64>) -> ()
  }) : () -> ()
}) : () -> ()

