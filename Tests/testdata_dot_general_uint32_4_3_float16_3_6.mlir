"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x6xf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x3xui32>, tensor<3x6xf16>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x6xf16>
    %5 = "stablehlo.convert"(%3#0) : (tensor<4x3xui32>) -> tensor<4x3xf16>
    %6 = "stablehlo.convert"(%3#1) : (tensor<3x6xf16>) -> tensor<3x6xf16>
    %7 = "stablehlo.dot_general"(%5, %6) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>}> : (tensor<4x3xf16>, tensor<3x6xf16>) -> tensor<4x6xf16>
    "stablehlo.custom_call"(%7, %4) <{call_target_name = "check.expect_almost_eq", has_side_effect = true}> : (tensor<4x6xf16>, tensor<4x6xf16>) -> ()
    "func.return"(%7) : (tensor<4x6xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x3xui32>, tensor<3x6xf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[4, 4, 0], [0, 3, 0], [1, 0, 0], [2, 3, 6]]> : tensor<4x3xui32>}> : () -> tensor<4x3xui32>
    %2 = "stablehlo.constant"() <{value = dense<[[-2.898440e+00, 1.852540e+00, -3.498050e+00, -1.629880e+00, -7.958980e-01, -4.046880e+00], [1.198240e+00, -2.576170e+00, -3.648440e+00, -1.153320e+00, -2.929690e+00, -5.292970e-01], [-7.308590e+00, 2.611330e+00, 3.431640e+00, 9.500000e+00, 4.707030e+00, 2.431640e+00]]> : tensor<3x6xf16>}> : () -> tensor<3x6xf16>
    "func.return"(%1, %2) : (tensor<4x3xui32>, tensor<3x6xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-6.800780e+00, -2.894530e+00, -2.859380e+01, -1.113280e+01, -1.490630e+01, -1.831250e+01], [3.593750e+00, -7.726560e+00, -1.094530e+01, -3.460940e+00, -8.789060e+00, -1.587890e+00], [-2.898440e+00, 1.852540e+00, -3.498050e+00, -1.629880e+00, -7.958980e-01, -4.046880e+00], [-4.606250e+01, 1.164060e+01, 2.648440e+00, 5.028130e+01, 1.785940e+01, 4.906250e+00]]> : tensor<4x6xf16>}> : () -> tensor<4x6xf16>
    "func.return"(%0) : (tensor<4x6xf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

