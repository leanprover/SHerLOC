"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2xcomplex<f64>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<2xcomplex<f64>>, tensor<2xcomplex<f64>>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<2xcomplex<f64>>
    %5 = "stablehlo.add"(%3#0, %3#1) : (tensor<2xcomplex<f64>>, tensor<2xcomplex<f64>>) -> tensor<2xcomplex<f64>>
    "stablehlo.custom_call"(%5, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<2xcomplex<f64>>, tensor<2xcomplex<f64>>) -> ()
    "func.return"(%5) : (tensor<2xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<2xcomplex<f64>>, tensor<2xcomplex<f64>>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[(0.42742970398722779,0.19525882468777409), (-1.5782763149646164,-4.0015414451742641)]> : tensor<2xcomplex<f64>>}> : () -> tensor<2xcomplex<f64>>
    %2 = "stablehlo.constant"() <{value = dense<[(-0.11290775549527821,-3.1382975860447448), (2.7883747974901301,-1.8044923462614515)]> : tensor<2xcomplex<f64>>}> : () -> tensor<2xcomplex<f64>>
    "func.return"(%1, %2) : (tensor<2xcomplex<f64>>, tensor<2xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2xcomplex<f64>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[(0.31452194849194959,-2.9430387613569708), (1.2100984825255137,-5.8060337914357154)]> : tensor<2xcomplex<f64>>}> : () -> tensor<2xcomplex<f64>>
    "func.return"(%0) : (tensor<2xcomplex<f64>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

