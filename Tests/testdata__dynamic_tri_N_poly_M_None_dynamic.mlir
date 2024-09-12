"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x1xf32>) -> tensor<?x?xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x1xf32>):
    %0 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %1 = "stablehlo.reshape"(%0) : (tensor<i32>) -> tensor<1xi32>
    %2 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %3 = "stablehlo.reshape"(%2) : (tensor<i32>) -> tensor<1xi32>
    %4 = "stablehlo.concatenate"(%1, %3) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %5 = "stablehlo.dynamic_iota"(%4) <{iota_dimension = 0 : i64}> : (tensor<2xi32>) -> tensor<?x?xi32>
    %6 = "stablehlo.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
    %7 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %8 = "stablehlo.reshape"(%7) : (tensor<i32>) -> tensor<1xi32>
    %9 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %10 = "stablehlo.reshape"(%9) : (tensor<i32>) -> tensor<1xi32>
    %11 = "stablehlo.concatenate"(%8, %10) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %12 = "stablehlo.dynamic_broadcast_in_dim"(%6, %11) <{broadcast_dimensions = array<i64>}> : (tensor<i32>, tensor<2xi32>) -> tensor<?x?xi32>
    %13 = "stablehlo.add"(%5, %12) : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
    %14 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %15 = "stablehlo.reshape"(%14) : (tensor<i32>) -> tensor<1xi32>
    %16 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %17 = "stablehlo.reshape"(%16) : (tensor<i32>) -> tensor<1xi32>
    %18 = "stablehlo.concatenate"(%15, %17) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %19 = "stablehlo.dynamic_iota"(%18) <{iota_dimension = 1 : i64}> : (tensor<2xi32>) -> tensor<?x?xi32>
    %20 = "stablehlo.compare"(%13, %19) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction GE>}> : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi1>
    %21 = "stablehlo.convert"(%20) : (tensor<?x?xi1>) -> tensor<?x?xf32>
    %22 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %23 = "stablehlo.reshape"(%22) : (tensor<i32>) -> tensor<1xi32>
    %24 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %25 = "stablehlo.reshape"(%24) : (tensor<i32>) -> tensor<1xi32>
    %26 = "stablehlo.concatenate"(%23, %25) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %27 = "stablehlo.dynamic_broadcast_in_dim"(%arg1, %26) <{broadcast_dimensions = array<i64: 0, 1>}> : (tensor<?x1xf32>, tensor<2xi32>) -> tensor<?x?xf32>
    %28 = "stablehlo.add"(%21, %27) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    "func.return"(%28) : (tensor<?x?xf32>) -> ()
  }) : () -> ()
}) : () -> ()

