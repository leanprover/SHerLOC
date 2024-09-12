"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x2x3xbf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[3, 2]> : tensor<2xi64>}> : () -> tensor<2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x2x3xbf16>, tensor<2xbf16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<4x2x3xbf16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [0, 2], scatter_dims_to_operand_dims = [0, 2]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %7 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<bf16>, tensor<bf16>) -> tensor<bf16>
      "stablehlo.return"(%7) : (tensor<bf16>) -> ()
    }) : (tensor<4x2x3xbf16>, tensor<2xi64>, tensor<2xbf16>) -> tensor<4x2x3xbf16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x2x3xbf16>, tensor<4x2x3xbf16>) -> ()
    "func.return"(%6) : (tensor<4x2x3xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x2x3xbf16>, tensor<2xbf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[[4.343750e+00, -4.687500e+00, -3.406250e+00], [5.117190e-01, -2.796880e+00, 3.554690e-01]], [[4.625000e+00, 1.375000e+00, 2.328130e+00], [-1.976560e+00, -3.500000e+00, -3.750000e+00]], [[-4.906250e+00, -1.914060e+00, -2.250000e+00], [-5.625000e+00, -3.578130e+00, 1.476560e+00]], [[2.312500e+00, -4.003910e-01, -2.437500e+00], [-2.687500e+00, -1.906250e+00, -4.343750e+00]]]> : tensor<4x2x3xbf16>}> : () -> tensor<4x2x3xbf16>
    %2 = "stablehlo.constant"() <{value = dense<[9.023430e-01, -4.125000e+00]> : tensor<2xbf16>}> : () -> tensor<2xbf16>
    "func.return"(%1, %2) : (tensor<4x2x3xbf16>, tensor<2xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x2x3xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[4.343750e+00, -4.687500e+00, -3.406250e+00], [5.117190e-01, -2.796880e+00, 3.554690e-01]], [[4.625000e+00, 1.375000e+00, 2.328130e+00], [-1.976560e+00, -3.500000e+00, -3.750000e+00]], [[-4.906250e+00, -1.914060e+00, -2.250000e+00], [-5.625000e+00, -3.578130e+00, 1.476560e+00]], [[2.312500e+00, -4.003910e-01, -2.203130e+00], [-2.687500e+00, -1.906250e+00, 1.787500e+01]]]> : tensor<4x2x3xbf16>}> : () -> tensor<4x2x3xbf16>
    "func.return"(%0) : (tensor<4x2x3xbf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

