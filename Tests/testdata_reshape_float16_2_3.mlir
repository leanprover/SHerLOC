"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x2xf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<2x3xf16>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<3x2xf16>
    %4 = "stablehlo.reshape"(%2) : (tensor<2x3xf16>) -> tensor<3x2xf16>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3x2xf16>, tensor<3x2xf16>) -> ()
    "func.return"(%4) : (tensor<3x2xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-4.937500e+00, 3.582030e+00, -5.273440e-01], [-7.766720e-03, 1.227540e+00, -7.031250e-01]]> : tensor<2x3xf16>}> : () -> tensor<2x3xf16>
    "func.return"(%1) : (tensor<2x3xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x2xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-4.937500e+00, 3.582030e+00], [-5.273440e-01, -7.766720e-03], [1.227540e+00, -7.031250e-01]]> : tensor<3x2xf16>}> : () -> tensor<3x2xf16>
    "func.return"(%0) : (tensor<3x2xf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

