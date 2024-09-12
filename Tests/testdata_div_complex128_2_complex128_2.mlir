"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2xcomplex<f64>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<2xcomplex<f64>>, tensor<2xcomplex<f64>>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<2xcomplex<f64>>
    %5 = "stablehlo.divide"(%3#0, %3#1) : (tensor<2xcomplex<f64>>, tensor<2xcomplex<f64>>) -> tensor<2xcomplex<f64>>
    "stablehlo.custom_call"(%5, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<2xcomplex<f64>>, tensor<2xcomplex<f64>>) -> ()
    "func.return"(%5) : (tensor<2xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<2xcomplex<f64>>, tensor<2xcomplex<f64>>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[(-0.4142508652598253,2.9708064697134127), (-0.086255762317640316,-2.1008513159963607)]> : tensor<2xcomplex<f64>>}> : () -> tensor<2xcomplex<f64>>
    %2 = "stablehlo.constant"() <{value = dense<[(-2.3701009044089654,3.734864852692362), (1.6009430068468937,-0.20405409383852913)]> : tensor<2xcomplex<f64>>}> : () -> tensor<2xcomplex<f64>>
    "func.return"(%1, %2) : (tensor<2xcomplex<f64>>, tensor<2xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2xcomplex<f64>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[(0.61724473678798175,-0.28078163066250666), (0.11156816366298221,-1.2980383227706118)]> : tensor<2xcomplex<f64>>}> : () -> tensor<2xcomplex<f64>>
    "func.return"(%0) : (tensor<2xcomplex<f64>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

