"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x3xbf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %5:4 = "func.call"() <{callee = @inputs}> : () -> (tensor<2x3xi32>, tensor<2x3xbf16>, tensor<2x3xbf16>, tensor<2x3xbf16>)
    %6 = "func.call"() <{callee = @expected}> : () -> tensor<2x3xbf16>
    %7 = "stablehlo.constant"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
    %8 = "stablehlo.broadcast_in_dim"(%7) <{broadcast_dimensions = array<i64>}> : (tensor<i32>) -> tensor<2x3xi32>
    %9 = "stablehlo.compare"(%5#0, %8) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi1>
    %10 = "stablehlo.constant"() <{value = dense<2> : tensor<i32>}> : () -> tensor<i32>
    %11 = "stablehlo.broadcast_in_dim"(%10) <{broadcast_dimensions = array<i64>}> : (tensor<i32>) -> tensor<2x3xi32>
    %12 = "stablehlo.compare"(%5#0, %11) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi1>
    %13 = "stablehlo.select"(%12, %5#2, %5#3) : (tensor<2x3xi1>, tensor<2x3xbf16>, tensor<2x3xbf16>) -> tensor<2x3xbf16>
    %14 = "stablehlo.select"(%9, %5#1, %13) : (tensor<2x3xi1>, tensor<2x3xbf16>, tensor<2x3xbf16>) -> tensor<2x3xbf16>
    "stablehlo.custom_call"(%14, %6) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<2x3xbf16>, tensor<2x3xbf16>) -> ()
    "func.return"(%14) : (tensor<2x3xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<2x3xi32>, tensor<2x3xbf16>, tensor<2x3xbf16>, tensor<2x3xbf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[0, 2, 1], [2, 0, 1]]> : tensor<2x3xi32>}> : () -> tensor<2x3xi32>
    %2 = "stablehlo.constant"() <{value = dense<[[-1.570310e+00, 2.243040e-03, -5.078130e-01], [-2.375000e+00, -1.609380e+00, -4.187500e+00]]> : tensor<2x3xbf16>}> : () -> tensor<2x3xbf16>
    %3 = "stablehlo.constant"() <{value = dense<[[-4.687500e+00, 3.156250e+00, -2.890630e+00], [2.687500e+00, 4.562500e+00, 3.296880e+00]]> : tensor<2x3xbf16>}> : () -> tensor<2x3xbf16>
    %4 = "stablehlo.constant"() <{value = dense<[[1.742190e+00, -2.640630e+00, 1.164060e+00], [-2.171880e+00, 2.265630e-01, 3.734380e+00]]> : tensor<2x3xbf16>}> : () -> tensor<2x3xbf16>
    "func.return"(%1, %2, %3, %4) : (tensor<2x3xi32>, tensor<2x3xbf16>, tensor<2x3xbf16>, tensor<2x3xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-1.570310e+00, -2.640630e+00, -2.890630e+00], [-2.171880e+00, -1.609380e+00, 3.296880e+00]]> : tensor<2x3xbf16>}> : () -> tensor<2x3xbf16>
    "func.return"(%0) : (tensor<2x3xbf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

