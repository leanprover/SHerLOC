"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<8x9xf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %5 = "func.call"() <{callee = @inputs}> : () -> tensor<8x9xf16>
    %6 = "func.call"() <{callee = @expected}> : () -> tensor<8x9xf16>
    %7 = "func.call"(%5) <{callee = @cumprod}> : (tensor<8x9xf16>) -> tensor<8x9xf16>
    "stablehlo.custom_call"(%7, %6) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<8x9xf16>, tensor<8x9xf16>) -> ()
    "func.return"(%7) : (tensor<8x9xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<8x9xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %4 = "stablehlo.constant"() <{value = dense<[[1.173830e+00, -1.326170e+00, -1.215210e-01, 1.446290e+00, -5.171880e+00, 3.423830e+00, 1.404300e+00, -6.791990e-01, -2.884770e+00], [2.791020e+00, 1.597660e+00, 3.333980e+00, -3.927730e+00, 6.650390e-01, 3.353520e+00, 3.855470e+00, 4.183590e+00, 1.718750e-01], [1.131840e+00, 4.906250e+00, -2.980960e-01, -3.812500e+00, 2.076170e+00, 3.413090e-01, -4.039060e+00, -5.599980e-02, 2.251950e+00], [-3.892580e+00, -4.777340e+00, -4.042970e+00, 2.732420e+00, 1.374020e+00, 1.184570e+00, 8.764640e-01, -1.516600e+00, 3.076170e+00], [-4.023440e+00, -3.480470e+00, 2.451170e+00, -6.281250e+00, 7.890630e+00, 3.195310e+00, 2.197270e+00, 2.603520e+00, -3.191410e+00], [4.347660e+00, 6.938470e-01, -2.207030e+00, 1.920900e+00, -2.597660e+00, 1.887700e+00, 4.855470e+00, 2.001950e+00, 1.344730e+00], [5.332030e+00, 1.693360e+00, -2.484380e+00, 1.912110e+00, 5.511710e+00, -4.765630e+00, -1.393130e-02, -4.621580e-01, -2.238280e+00], [2.343750e+00, -2.681640e+00, 2.007810e+00, -2.574220e+00, -4.199220e-01, -2.029300e+00, -6.098630e-01, -1.702150e+00, -2.193360e+00]]> : tensor<8x9xf16>}> : () -> tensor<8x9xf16>
    "func.return"(%4) : (tensor<8x9xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<8x9xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[1.173830e+00, -1.326170e+00, -1.215210e-01, 1.446290e+00, -5.171880e+00, 3.423830e+00, 1.404300e+00, -6.791990e-01, -2.884770e+00], [3.275390e+00, -2.119140e+00, -4.050290e-01, -5.679690e+00, -3.439450e+00, 1.148440e+01, 5.414060e+00, -2.841800e+00, -4.958500e-01], [3.707030e+00, -1.039840e+01, 1.207280e-01, 2.165630e+01, -7.140630e+00, 3.919920e+00, -2.187500e+01, 1.591800e-01, -1.116210e+00], [-1.442970e+01, 4.968750e+01, -4.880370e-01, 5.918750e+01, -9.812500e+00, 4.644530e+00, -1.917190e+01, -2.414550e-01, -3.433590e+00], [5.806250e+01, -1.728750e+02, -1.196290e+00, -3.717500e+02, -7.743750e+01, 1.484380e+01, -4.212500e+01, -6.284180e-01, 1.096090e+01], [2.523750e+02, -1.199380e+02, 2.640630e+00, -7.140000e+02, 2.011250e+02, 2.801560e+01, -2.045000e+02, -1.257810e+00, 1.474220e+01], [1.346000e+03, -2.031250e+02, -6.558590e+00, -1.365000e+03, 1.109000e+03, -1.335000e+02, 2.849610e+00, 5.815430e-01, -3.300000e+01], [3.154000e+03, 5.445000e+02, -1.317190e+01, 3.514000e+03, -4.657500e+02, 2.710000e+02, -1.738280e+00, -9.897460e-01, 7.237500e+01]]> : tensor<8x9xf16>}> : () -> tensor<8x9xf16>
    "func.return"(%3) : (tensor<8x9xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<8x9xf16>) -> tensor<8x9xf16>, sym_name = "cumprod", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<8x9xf16>):
    %0 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f16>}> : () -> tensor<f16>
    %1 = "stablehlo.reduce_window"(%arg0, %0) <{padding = dense<[[7, 0], [0, 0]]> : tensor<2x2xi64>, window_dimensions = array<i64: 8, 1>}> ({
    ^bb0(%arg1: tensor<f16>, %arg2: tensor<f16>):
      %2 = "stablehlo.multiply"(%arg1, %arg2) : (tensor<f16>, tensor<f16>) -> tensor<f16>
      "stablehlo.return"(%2) : (tensor<f16>) -> ()
    }) : (tensor<8x9xf16>, tensor<f16>) -> tensor<8x9xf16>
    "func.return"(%1) : (tensor<8x9xf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

