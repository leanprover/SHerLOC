"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x3xbf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %4:3 = "func.call"() <{callee = @inputs}> : () -> (tensor<bf16>, tensor<2x3xbf16>, tensor<bf16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<2x3xbf16>
    %6 = "stablehlo.broadcast_in_dim"(%4#0) <{broadcast_dimensions = array<i64>}> : (tensor<bf16>) -> tensor<2x3xbf16>
    %7 = "stablehlo.broadcast_in_dim"(%4#2) <{broadcast_dimensions = array<i64>}> : (tensor<bf16>) -> tensor<2x3xbf16>
    %8 = "stablehlo.clamp"(%6, %4#1, %7) : (tensor<2x3xbf16>, tensor<2x3xbf16>, tensor<2x3xbf16>) -> tensor<2x3xbf16>
    "stablehlo.custom_call"(%8, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<2x3xbf16>, tensor<2x3xbf16>) -> ()
    "func.return"(%8) : (tensor<2x3xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<bf16>, tensor<2x3xbf16>, tensor<bf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-4.406250e+00, -2.953130e+00, -1.609380e+00], [6.054690e-01, -1.748050e-01, 6.953130e-01]]> : tensor<2x3xbf16>}> : () -> tensor<2x3xbf16>
    %2 = "stablehlo.constant"() <{value = dense<-5.859380e-01> : tensor<bf16>}> : () -> tensor<bf16>
    %3 = "stablehlo.constant"() <{value = dense<-3.906250e+00> : tensor<bf16>}> : () -> tensor<bf16>
    "func.return"(%2, %1, %3) : (tensor<bf16>, tensor<2x3xbf16>, tensor<bf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<-3.906250e+00> : tensor<2x3xbf16>}> : () -> tensor<2x3xbf16>
    "func.return"(%0) : (tensor<2x3xbf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

