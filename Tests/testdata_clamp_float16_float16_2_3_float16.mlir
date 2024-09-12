"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x3xf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %4:3 = "func.call"() <{callee = @inputs}> : () -> (tensor<f16>, tensor<2x3xf16>, tensor<f16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<2x3xf16>
    %6 = "stablehlo.broadcast_in_dim"(%4#0) <{broadcast_dimensions = array<i64>}> : (tensor<f16>) -> tensor<2x3xf16>
    %7 = "stablehlo.broadcast_in_dim"(%4#2) <{broadcast_dimensions = array<i64>}> : (tensor<f16>) -> tensor<2x3xf16>
    %8 = "stablehlo.clamp"(%6, %4#1, %7) : (tensor<2x3xf16>, tensor<2x3xf16>, tensor<2x3xf16>) -> tensor<2x3xf16>
    "stablehlo.custom_call"(%8, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<2x3xf16>, tensor<2x3xf16>) -> ()
    "func.return"(%8) : (tensor<2x3xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<f16>, tensor<2x3xf16>, tensor<f16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[2.397460e-01, 2.158200e+00, 2.296880e+00], [-2.714840e+00, -2.687500e+00, -1.233400e+00]]> : tensor<2x3xf16>}> : () -> tensor<2x3xf16>
    %2 = "stablehlo.constant"() <{value = dense<3.716800e+00> : tensor<f16>}> : () -> tensor<f16>
    %3 = "stablehlo.constant"() <{value = dense<-4.324220e+00> : tensor<f16>}> : () -> tensor<f16>
    "func.return"(%2, %1, %3) : (tensor<f16>, tensor<2x3xf16>, tensor<f16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<-4.324220e+00> : tensor<2x3xf16>}> : () -> tensor<2x3xf16>
    "func.return"(%0) : (tensor<2x3xf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

