"builtin.module"() <{sym_name = "jit_fun_flat_jax"}> ({
  "func.func"() <{arg_attrs = [{}, {mhlo.sharding = ""}, {mhlo.sharding = ""}], function_type = (tensor<i64>, tensor<?x2x3xbf16>, tensor<?xbf16>) -> tensor<?x2x3xbf16>, sym_name = "main", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<i64>, %arg1: tensor<?x2x3xbf16>, %arg2: tensor<?xbf16>):
    %0 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<bf16>}> : () -> tensor<bf16>
    %1 = "stablehlo.pad"(%arg1, %0) <{edge_padding_high = array<i64: 0, 0, 0>, edge_padding_low = array<i64: 0, 0, 0>, interior_padding = array<i64: 0, 0, 0>}> : (tensor<?x2x3xbf16>, tensor<bf16>) -> tensor<?x2x3xbf16>
    %2 = "stablehlo.constant"() <{value = dense<true> : tensor<i1>}> : () -> tensor<i1>
    %3 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %4 = "stablehlo.reshape"(%3) : (tensor<i32>) -> tensor<1xi32>
    %5 = "stablehlo.constant"() <{value = dense<2> : tensor<1xi32>}> : () -> tensor<1xi32>
    %6 = "stablehlo.constant"() <{value = dense<3> : tensor<1xi32>}> : () -> tensor<1xi32>
    %7 = "stablehlo.concatenate"(%4, %5, %6) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %8 = "stablehlo.dynamic_broadcast_in_dim"(%2, %7) <{broadcast_dimensions = array<i64>}> : (tensor<i1>, tensor<3xi32>) -> tensor<?x2x3xi1>
    %9 = "stablehlo.constant"() <{value = dense<false> : tensor<i1>}> : () -> tensor<i1>
    %10 = "stablehlo.pad"(%8, %9) <{edge_padding_high = array<i64: 0, 0, 0>, edge_padding_low = array<i64: 0, 0, 0>, interior_padding = array<i64: 0, 0, 0>}> : (tensor<?x2x3xi1>, tensor<i1>) -> tensor<?x2x3xi1>
    %11 = "stablehlo.convert"(%arg0) : (tensor<i64>) -> tensor<i32>
    %12 = "stablehlo.reshape"(%11) : (tensor<i32>) -> tensor<1xi32>
    %13 = "stablehlo.constant"() <{value = dense<2> : tensor<1xi32>}> : () -> tensor<1xi32>
    %14 = "stablehlo.constant"() <{value = dense<3> : tensor<1xi32>}> : () -> tensor<1xi32>
    %15 = "stablehlo.concatenate"(%12, %13, %14) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %16 = "stablehlo.dynamic_broadcast_in_dim"(%arg2, %15) <{broadcast_dimensions = array<i64: 0>}> : (tensor<?xbf16>, tensor<3xi32>) -> tensor<?x2x3xbf16>
    %17 = "stablehlo.select"(%10, %1, %16) : (tensor<?x2x3xi1>, tensor<?x2x3xbf16>, tensor<?x2x3xbf16>) -> tensor<?x2x3xbf16>
    "func.return"(%17) : (tensor<?x2x3xbf16>) -> ()
  }) : () -> ()
}) : () -> ()

