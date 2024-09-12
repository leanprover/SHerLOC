"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x20x20xi32>, tensor<?x1x20xi32>) -> tensor<?x20x20xi32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x20x20xi32>, %arg2: tensor<?x1x20xi32>):
    %0 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %1 = "stablehlo.reshape"(%0) : (tensor<i32>) -> tensor<1xi32>
    %2 = "stablehlo.constant"() <{value = dense<20> : tensor<1xi32>}> : () -> tensor<1xi32>
    %3 = "stablehlo.constant"() <{value = dense<20> : tensor<1xi32>}> : () -> tensor<1xi32>
    %4 = "stablehlo.concatenate"(%1, %2, %3) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %5 = "stablehlo.dynamic_broadcast_in_dim"(%arg2, %4) <{broadcast_dimensions = array<i64: 0, 1, 2>}> : (tensor<?x1x20xi32>, tensor<3xi32>) -> tensor<?x20x20xi32>
    %6 = "stablehlo.and"(%arg1, %5) : (tensor<?x20x20xi32>, tensor<?x20x20xi32>) -> tensor<?x20x20xi32>
    "func.return"(%6) : (tensor<?x20x20xi32>) -> ()
  }) : () -> ()
}) : () -> ()

