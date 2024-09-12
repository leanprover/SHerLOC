"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x1xf32>) -> tensor<?x1xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x1xf32>):
    %0 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %1 = "stablehlo.reshape"(%0) : (tensor<i32>) -> tensor<1xi32>
    %2 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %3 = "stablehlo.concatenate"(%1, %2) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %4 = "stablehlo.dynamic_iota"(%3) <{iota_dimension = 0 : i64}> : (tensor<2xi32>) -> tensor<?x1xui32>
    %5 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %6 = "stablehlo.reshape"(%5) : (tensor<i32>) -> tensor<1xi32>
    %7 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
    %8 = "stablehlo.concatenate"(%6, %7) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %9 = "stablehlo.dynamic_iota"(%8) <{iota_dimension = 1 : i64}> : (tensor<2xi32>) -> tensor<?x1xui32>
    %10 = "stablehlo.compare"(%4, %9) <{compare_type = #stablehlo<comparison_type UNSIGNED>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<?x1xui32>, tensor<?x1xui32>) -> tensor<?x1xi1>
    %11 = "stablehlo.convert"(%10) : (tensor<?x1xi1>) -> tensor<?x1xf32>
    %12 = "stablehlo.add"(%11, %arg1) : (tensor<?x1xf32>, tensor<?x1xf32>) -> tensor<?x1xf32>
    "func.return"(%12) : (tensor<?x1xf32>) -> ()
  }) : () -> ()
}) : () -> ()

