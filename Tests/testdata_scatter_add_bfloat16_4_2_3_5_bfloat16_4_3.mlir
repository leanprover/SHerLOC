"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x2x3x5xbf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[0, 4]> : tensor<2xi64>}> : () -> tensor<2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x2x3x5xbf16>, tensor<4x3xbf16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<4x2x3x5xbf16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %7 = "stablehlo.add"(%arg0, %arg1) : (tensor<bf16>, tensor<bf16>) -> tensor<bf16>
      "stablehlo.return"(%7) : (tensor<bf16>) -> ()
    }) : (tensor<4x2x3x5xbf16>, tensor<2xi64>, tensor<4x3xbf16>) -> tensor<4x2x3x5xbf16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x2x3x5xbf16>, tensor<4x2x3x5xbf16>) -> ()
    "func.return"(%6) : (tensor<4x2x3x5xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x2x3x5xbf16>, tensor<4x3xbf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x97C08C40AFBF123F9FC0A1C01D40BEBFF8BF393F693F58C083400140AF4051BF40BE7C40813F4AC059404F40C73FC33FE5BFEDC0E6BF824049C09D40343F3F3EF9BFD83EE73F7BBF25409B3F9DBC853E4C40124017C019C050C01CC021BF7EBF8DBE25C043BDAC4062C045C012C0933FA73F01C091BFC2C0B13F333E9FC004C0B740E03FA43FAFC01E402A40103F97C0CD3FAABF0FC022409AC085C0E5401040913ECC3EDD3F94C03D40B7C054402FC02BBF10C0AF3F7CC011BFC4BF4AC06AC04EC04DBFCEC056C021C045400440DABFC03FD0405EC02D4076BFD83F83C083BF7BC005C01FC035C07D4044BF89BF7BC0"> : tensor<4x2x3x5xbf16>}> : () -> tensor<4x2x3x5xbf16>
    %2 = "stablehlo.constant"() <{value = dense<[[2.375000e+00, 3.421880e+00, 9.101560e-01], [-2.031250e+00, -6.218750e+00, 2.093750e+00], [1.382810e+00, -1.242190e+00, -2.375000e+00], [-2.656250e+00, -6.953130e-01, 3.496090e-01]]> : tensor<4x3xbf16>}> : () -> tensor<4x3xbf16>
    "func.return"(%1, %2) : (tensor<4x2x3x5xbf16>, tensor<4x3xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x2x3x5xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x97C08C40AFBF123F26C0A1C01D40BEBFF8BF8540693F58C083400140CC4051BF40BE7C40813F4AC059404F40C73FC33FE5BFEDC0E6BF824049C09D40343F3F3EF9BFD83E68BE7BBF25409B3F9DBCBFC04C40124017C019C094BF1CC021BF7EBF8DBE25C043BDAC4062C045C012C0933FA73F01C091BFC2C0B13F333E9FC004C0E340E03FA43FAFC01E40B53F103F97C0CD3FAABF94C022409AC085C0E5401040913ECC3EDD3F94C03D40B7C054402FC02BBF10C0AF3F7CC011BFC4BFBAC06AC04EC04DBFCEC081C021C045400440DABFED3FD0405EC02D4076BFD83F83C083BF7BC005C01FC035C07D4044BF89BF7BC0"> : tensor<4x2x3x5xbf16>}> : () -> tensor<4x2x3x5xbf16>
    "func.return"(%0) : (tensor<4x2x3x5xbf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

