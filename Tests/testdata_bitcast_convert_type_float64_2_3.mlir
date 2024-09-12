"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x3xui64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<2x3xf64>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<2x3xui64>
    %4 = "stablehlo.bitcast_convert"(%2) : (tensor<2x3xf64>) -> tensor<2x3xui64>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<2x3xui64>, tensor<2x3xui64>) -> ()
    "func.return"(%4) : (tensor<2x3xui64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[2.8308635969433493, 0.71900016032753111, -1.6188963883466747], [-2.954063914647203, 0.26588954076144977, -0.30983069279339981]]> : tensor<2x3xf64>}> : () -> tensor<2x3xf64>
    "func.return"(%1) : (tensor<2x3xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xui64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[4613556956920182793, 4604651397253537208, 13833341717198732246], [13837206416227410102, 4598461460064685830, 13822625070343130920]]> : tensor<2x3xui64>}> : () -> tensor<2x3xui64>
    "func.return"(%0) : (tensor<2x3xui64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

