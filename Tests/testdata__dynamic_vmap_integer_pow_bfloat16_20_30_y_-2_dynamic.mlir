"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x20x30xbf16>) -> tensor<?x20x30xbf16>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x20x30xbf16>):
    %0 = "stablehlo.multiply"(%arg1, %arg1) : (tensor<?x20x30xbf16>, tensor<?x20x30xbf16>) -> tensor<?x20x30xbf16>
    %1 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<bf16>}> : () -> tensor<bf16>
    %2 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %3 = "stablehlo.reshape"(%2) : (tensor<i32>) -> tensor<1xi32>
    %4 = "stablehlo.constant"() <{value = dense<20> : tensor<1xi32>}> : () -> tensor<1xi32>
    %5 = "stablehlo.constant"() <{value = dense<30> : tensor<1xi32>}> : () -> tensor<1xi32>
    %6 = "stablehlo.concatenate"(%3, %4, %5) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %7 = "stablehlo.dynamic_broadcast_in_dim"(%1, %6) <{broadcast_dimensions = array<i64>}> : (tensor<bf16>, tensor<3xi32>) -> tensor<?x20x30xbf16>
    %8 = "stablehlo.divide"(%7, %0) : (tensor<?x20x30xbf16>, tensor<?x20x30xbf16>) -> tensor<?x20x30xbf16>
    "func.return"(%8) : (tensor<?x20x30xbf16>) -> ()
  }) : () -> ()
}) : () -> ()

