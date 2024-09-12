"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<8x9xf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %6 = "func.call"() <{callee = @inputs}> : () -> tensor<8x9xf16>
    %7 = "func.call"() <{callee = @expected}> : () -> tensor<8x9xf16>
    %8 = "func.call"(%6) <{callee = @cummax}> : (tensor<8x9xf16>) -> tensor<8x9xf16>
    "stablehlo.custom_call"(%8, %7) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<8x9xf16>, tensor<8x9xf16>) -> ()
    "func.return"(%8) : (tensor<8x9xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<8x9xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %5 = "stablehlo.constant"() <{value = dense<[[-6.186520e-01, 3.033200e+00, 2.978520e+00, -4.535160e+00, -3.171880e+00, -5.035160e+00, -2.421880e+00, -3.779300e+00, -2.730470e+00], [3.623050e-01, 7.143550e-01, -4.710940e+00, -4.199220e-01, 2.601560e+00, -9.887690e-01, 8.403320e-01, 1.086430e-01, 2.517090e-01], [-3.406250e+00, -3.289060e+00, -1.094730e+00, -5.014650e-01, -2.966800e+00, 5.229490e-01, 1.628910e+00, 1.155270e+00, -2.535160e+00], [-2.736330e+00, -6.425780e-01, -9.780270e-01, 4.855470e+00, -1.765630e+00, -1.186520e+00, 8.728020e-02, -2.748050e+00, -5.414060e+00], [-1.822270e+00, 2.070310e+00, -6.689450e-02, -4.614260e-01, 2.667970e+00, 1.663090e+00, 3.801270e-01, 1.887700e+00, 4.175780e+00], [1.050780e+00, -7.335930e+00, -1.176760e+00, 3.509770e+00, -2.902340e+00, 1.328130e+00, -4.062500e+00, -1.000980e+00, -1.870120e+00], [-9.619140e-01, 9.355460e-01, -5.166020e-01, 2.509770e+00, 5.041500e-02, 3.617190e+00, 1.008790e+00, 2.894530e+00, -2.992190e+00], [1.180660e+00, -1.629880e+00, 1.940430e+00, -3.410160e+00, -7.640630e+00, 6.352540e-01, -2.593990e-02, -5.308590e+00, 2.566410e+00]]> : tensor<8x9xf16>}> : () -> tensor<8x9xf16>
    "func.return"(%5) : (tensor<8x9xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<8x9xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %4 = "stablehlo.constant"() <{value = dense<[[-6.186520e-01, 3.033200e+00, 2.978520e+00, -4.535160e+00, -3.171880e+00, -5.035160e+00, -2.421880e+00, -3.779300e+00, -2.730470e+00], [3.623050e-01, 3.033200e+00, 2.978520e+00, -4.199220e-01, 2.601560e+00, -9.887690e-01, 8.403320e-01, 1.086430e-01, 2.517090e-01], [3.623050e-01, 3.033200e+00, 2.978520e+00, -4.199220e-01, 2.601560e+00, 5.229490e-01, 1.628910e+00, 1.155270e+00, 2.517090e-01], [3.623050e-01, 3.033200e+00, 2.978520e+00, 4.855470e+00, 2.601560e+00, 5.229490e-01, 1.628910e+00, 1.155270e+00, 2.517090e-01], [3.623050e-01, 3.033200e+00, 2.978520e+00, 4.855470e+00, 2.667970e+00, 1.663090e+00, 1.628910e+00, 1.887700e+00, 4.175780e+00], [1.050780e+00, 3.033200e+00, 2.978520e+00, 4.855470e+00, 2.667970e+00, 1.663090e+00, 1.628910e+00, 1.887700e+00, 4.175780e+00], [1.050780e+00, 3.033200e+00, 2.978520e+00, 4.855470e+00, 2.667970e+00, 3.617190e+00, 1.628910e+00, 2.894530e+00, 4.175780e+00], [1.180660e+00, 3.033200e+00, 2.978520e+00, 4.855470e+00, 2.667970e+00, 3.617190e+00, 1.628910e+00, 2.894530e+00, 4.175780e+00]]> : tensor<8x9xf16>}> : () -> tensor<8x9xf16>
    "func.return"(%4) : (tensor<8x9xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<8x9xf16>) -> tensor<8x9xf16>, sym_name = "cummax", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<8x9xf16>):
    %0 = "stablehlo.constant"() <{value = dense<0xFC00> : tensor<f16>}> : () -> tensor<f16>
    %1 = "stablehlo.broadcast_in_dim"(%0) <{broadcast_dimensions = array<i64>}> : (tensor<f16>) -> tensor<f16>
    %2 = "stablehlo.reduce_window"(%arg0, %1) <{padding = dense<[[7, 0], [0, 0]]> : tensor<2x2xi64>, window_dimensions = array<i64: 8, 1>}> ({
    ^bb0(%arg1: tensor<f16>, %arg2: tensor<f16>):
      %3 = "stablehlo.maximum"(%arg1, %arg2) : (tensor<f16>, tensor<f16>) -> tensor<f16>
      "stablehlo.return"(%3) : (tensor<f16>) -> ()
    }) : (tensor<8x9xf16>, tensor<f16>) -> tensor<8x9xf16>
    "func.return"(%2) : (tensor<8x9xf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

