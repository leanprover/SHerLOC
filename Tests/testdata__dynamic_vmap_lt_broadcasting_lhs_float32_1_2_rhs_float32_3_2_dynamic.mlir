"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x1x2xf32>, tensor<?x3x2xf32>) -> tensor<?x3x2xi1>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x1x2xf32>, %arg2: tensor<?x3x2xf32>):
    %0 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %1 = "stablehlo.reshape"(%0) : (tensor<i32>) -> tensor<1xi32>
    %2 = "stablehlo.constant"() <{value = dense<3> : tensor<1xi32>}> : () -> tensor<1xi32>
    %3 = "stablehlo.constant"() <{value = dense<2> : tensor<1xi32>}> : () -> tensor<1xi32>
    %4 = "stablehlo.concatenate"(%1, %2, %3) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %5 = "stablehlo.dynamic_broadcast_in_dim"(%arg1, %4) <{broadcast_dimensions = array<i64: 0, 1, 2>}> : (tensor<?x1x2xf32>, tensor<3xi32>) -> tensor<?x3x2xf32>
    %6 = "stablehlo.compare"(%5, %arg2) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<?x3x2xf32>, tensor<?x3x2xf32>) -> tensor<?x3x2xi1>
    "func.return"(%6) : (tensor<?x3x2xi1>) -> ()
  }) : () -> ()
}) : () -> ()

