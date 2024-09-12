"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x4xf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<3x4xf16>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<3x4xf16>
    "stablehlo.custom_call"(%2, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3x4xf16>, tensor<3x4xf16>) -> ()
    "func.return"(%2) : (tensor<3x4xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x4xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[9.433590e-01, -1.404300e+00, -4.335940e-01, -4.843750e+00], [-2.107420e+00, -1.970700e+00, -7.065420e-01, -9.868160e-01], [-7.929680e+00, 2.261720e+00, 3.748050e+00, 1.697270e+00]]> : tensor<3x4xf16>}> : () -> tensor<3x4xf16>
    "func.return"(%1) : (tensor<3x4xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x4xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[9.433590e-01, -1.404300e+00, -4.335940e-01, -4.843750e+00], [-2.107420e+00, -1.970700e+00, -7.065420e-01, -9.868160e-01], [-7.929680e+00, 2.261720e+00, 3.748050e+00, 1.697270e+00]]> : tensor<3x4xf16>}> : () -> tensor<3x4xf16>
    "func.return"(%0) : (tensor<3x4xf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

