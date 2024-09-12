"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x20x20xbf16>) -> tensor<?x20x20xbf16>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x20x20xbf16>):
    %0 = "stablehlo.constant"() <{value = dense<20> : tensor<1xi32>}> : () -> tensor<1xi32>
    %1 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<bf16>}> : () -> tensor<bf16>
    %2 = "stablehlo.get_dimension_size"(%arg1) <{dimension = 0 : i64}> : (tensor<?x20x20xbf16>) -> tensor<i32>
    %3 = "stablehlo.reshape"(%2) : (tensor<i32>) -> tensor<1xi32>
    %4 = "stablehlo.concatenate"(%3, %0, %0) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %5 = "stablehlo.dynamic_broadcast_in_dim"(%1, %4) <{broadcast_dimensions = array<i64>}> : (tensor<bf16>, tensor<3xi32>) -> tensor<?x20x20xbf16>
    %6 = "stablehlo.atan2"(%arg1, %5) : (tensor<?x20x20xbf16>, tensor<?x20x20xbf16>) -> tensor<?x20x20xbf16>
    "func.return"(%6) : (tensor<?x20x20xbf16>) -> ()
  }) : () -> ()
}) : () -> ()

