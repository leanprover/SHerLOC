"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x4x5x6xf32>, tensor<?x4x1x6xf32>) -> tensor<?x4x5x6xf32>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x4x5x6xf32>, %arg2: tensor<?x4x1x6xf32>):
    %0 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %1 = "stablehlo.reshape"(%0) : (tensor<i32>) -> tensor<1xi32>
    %2 = "stablehlo.constant"() <{value = dense<4> : tensor<1xi32>}> : () -> tensor<1xi32>
    %3 = "stablehlo.constant"() <{value = dense<5> : tensor<1xi32>}> : () -> tensor<1xi32>
    %4 = "stablehlo.constant"() <{value = dense<6> : tensor<1xi32>}> : () -> tensor<1xi32>
    %5 = "stablehlo.concatenate"(%1, %2, %3, %4) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
    %6 = "stablehlo.dynamic_broadcast_in_dim"(%arg2, %5) <{broadcast_dimensions = array<i64: 0, 1, 2, 3>}> : (tensor<?x4x1x6xf32>, tensor<4xi32>) -> tensor<?x4x5x6xf32>
    %7 = "stablehlo.power"(%arg1, %6) : (tensor<?x4x5x6xf32>, tensor<?x4x5x6xf32>) -> tensor<?x4x5x6xf32>
    "func.return"(%7) : (tensor<?x4x5x6xf32>) -> ()
  }) : () -> ()
}) : () -> ()

