"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x5xf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<4x6xf16>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<3x5xf16>
    %4 = "stablehlo.constant"() <{value = dense<0x7C00> : tensor<f16>}> : () -> tensor<f16>
    %5 = "stablehlo.broadcast_in_dim"(%4) <{broadcast_dimensions = array<i64>}> : (tensor<f16>) -> tensor<f16>
    %6 = "stablehlo.reduce_window"(%2, %5) <{window_dimensions = array<i64: 2, 2>}> ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %7 = "stablehlo.minimum"(%arg0, %arg1) : (tensor<f16>, tensor<f16>) -> tensor<f16>
      "stablehlo.return"(%7) : (tensor<f16>) -> ()
    }) : (tensor<4x6xf16>, tensor<f16>) -> tensor<3x5xf16>
    "stablehlo.custom_call"(%6, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3x5xf16>, tensor<3x5xf16>) -> ()
    "func.return"(%6) : (tensor<3x5xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[4.085940e+00, 1.006840e+00, 1.763670e+00, -5.851560e+00, 5.908200e-01, -2.433590e+00], [-5.878900e+00, -2.517090e-01, 3.554690e+00, 4.136720e+00, -2.068360e+00, -1.481450e+00], [3.496090e+00, 1.714840e+00, -6.791990e-01, -7.260740e-01, 3.552730e+00, -2.517580e+00], [-6.269530e+00, 6.977530e-01, 4.808590e+00, 2.189450e+00, -6.234380e+00, 5.789060e+00]]> : tensor<4x6xf16>}> : () -> tensor<4x6xf16>
    "func.return"(%1) : (tensor<4x6xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x5xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-5.878900e+00, -2.517090e-01, -5.851560e+00, -5.851560e+00, -2.433590e+00], [-5.878900e+00, -6.791990e-01, -7.260740e-01, -2.068360e+00, -2.517580e+00], [-6.269530e+00, -6.791990e-01, -7.260740e-01, -6.234380e+00, -6.234380e+00]]> : tensor<3x5xf16>}> : () -> tensor<3x5xf16>
    "func.return"(%0) : (tensor<3x5xf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

