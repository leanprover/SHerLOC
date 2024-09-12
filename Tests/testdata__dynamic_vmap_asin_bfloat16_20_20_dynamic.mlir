"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x20x20xbf16>) -> tensor<?x20x20xbf16>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x20x20xbf16>):
    %0 = "stablehlo.constant"() <{value = dense<20> : tensor<1xi32>}> : () -> tensor<1xi32>
    %1 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<bf16>}> : () -> tensor<bf16>
    %2 = "stablehlo.constant"() <{value = dense<2.000000e+00> : tensor<bf16>}> : () -> tensor<bf16>
    %3 = "stablehlo.get_dimension_size"(%arg1) <{dimension = 0 : i64}> : (tensor<?x20x20xbf16>) -> tensor<i32>
    %4 = "stablehlo.reshape"(%3) : (tensor<i32>) -> tensor<1xi32>
    %5 = "stablehlo.concatenate"(%4, %0, %0) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %6 = "stablehlo.dynamic_broadcast_in_dim"(%2, %5) <{broadcast_dimensions = array<i64>}> : (tensor<bf16>, tensor<3xi32>) -> tensor<?x20x20xbf16>
    %7 = "stablehlo.get_dimension_size"(%arg1) <{dimension = 0 : i64}> : (tensor<?x20x20xbf16>) -> tensor<i32>
    %8 = "stablehlo.reshape"(%7) : (tensor<i32>) -> tensor<1xi32>
    %9 = "stablehlo.concatenate"(%8, %0, %0) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %10 = "stablehlo.dynamic_broadcast_in_dim"(%1, %9) <{broadcast_dimensions = array<i64>}> : (tensor<bf16>, tensor<3xi32>) -> tensor<?x20x20xbf16>
    %11 = "stablehlo.get_dimension_size"(%arg1) <{dimension = 0 : i64}> : (tensor<?x20x20xbf16>) -> tensor<i32>
    %12 = "stablehlo.reshape"(%11) : (tensor<i32>) -> tensor<1xi32>
    %13 = "stablehlo.concatenate"(%12, %0, %0) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %14 = "stablehlo.dynamic_broadcast_in_dim"(%1, %13) <{broadcast_dimensions = array<i64>}> : (tensor<bf16>, tensor<3xi32>) -> tensor<?x20x20xbf16>
    %15 = "stablehlo.multiply"(%arg1, %arg1) : (tensor<?x20x20xbf16>, tensor<?x20x20xbf16>) -> tensor<?x20x20xbf16>
    %16 = "stablehlo.subtract"(%14, %15) : (tensor<?x20x20xbf16>, tensor<?x20x20xbf16>) -> tensor<?x20x20xbf16>
    %17 = "stablehlo.sqrt"(%16) : (tensor<?x20x20xbf16>) -> tensor<?x20x20xbf16>
    %18 = "stablehlo.add"(%10, %17) : (tensor<?x20x20xbf16>, tensor<?x20x20xbf16>) -> tensor<?x20x20xbf16>
    %19 = "stablehlo.atan2"(%arg1, %18) : (tensor<?x20x20xbf16>, tensor<?x20x20xbf16>) -> tensor<?x20x20xbf16>
    %20 = "stablehlo.multiply"(%6, %19) : (tensor<?x20x20xbf16>, tensor<?x20x20xbf16>) -> tensor<?x20x20xbf16>
    "func.return"(%20) : (tensor<?x20x20xbf16>) -> ()
  }) : () -> ()
}) : () -> ()

