"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x6xbf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x3xi16>, tensor<3x6xbf16>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x6xbf16>
    %5 = "stablehlo.convert"(%3#0) : (tensor<4x3xi16>) -> tensor<4x3xbf16>
    %6 = "stablehlo.convert"(%3#1) : (tensor<3x6xbf16>) -> tensor<3x6xbf16>
    %7 = "stablehlo.dot_general"(%5, %6) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>}> : (tensor<4x3xbf16>, tensor<3x6xbf16>) -> tensor<4x6xbf16>
    "stablehlo.custom_call"(%7, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x6xbf16>, tensor<4x6xbf16>) -> ()
    "func.return"(%7) : (tensor<4x6xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x3xi16>, tensor<3x6xbf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[2, -1, -2], [-3, 3, 2], [0, -3, 0], [-2, 2, 0]]> : tensor<4x3xi16>}> : () -> tensor<4x3xi16>
    %2 = "stablehlo.constant"() <{value = dense<[[1.187500e+00, 5.906250e+00, 5.406250e+00, -5.898440e-01, 2.281250e+00, 1.552730e-01], [1.609380e+00, 1.273440e+00, -3.937500e+00, 6.562500e-01, 1.015630e+00, -1.777340e-01], [1.804690e+00, -2.406250e+00, -2.968750e-01, 3.015630e+00, -5.312500e+00, 2.828130e+00]]> : tensor<3x6xbf16>}> : () -> tensor<3x6xbf16>
    "func.return"(%1, %2) : (tensor<4x3xi16>, tensor<3x6xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-2.843750e+00, 1.537500e+01, 1.537500e+01, -7.875000e+00, 1.418750e+01, -5.156250e+00], [4.875000e+00, -1.875000e+01, -2.862500e+01, 9.750000e+00, -1.443750e+01, 4.656250e+00], [-4.812500e+00, -3.812500e+00, 1.181250e+01, -1.968750e+00, -3.046880e+00, 5.312500e-01], [8.437500e-01, -9.250000e+00, -1.875000e+01, 2.500000e+00, -2.531250e+00, -6.640630e-01]]> : tensor<4x6xbf16>}> : () -> tensor<4x6xbf16>
    "func.return"(%0) : (tensor<4x6xbf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

