"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x1x2xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<7x5x3xf32>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<3x1x2xf32>
    %4 = "stablehlo.slice"(%2) <{limit_indices = array<i64: 7, 1, 3>, start_indices = array<i64: 4, 0, 1>, strides = array<i64: 1, 1, 1>}> : (tensor<7x5x3xf32>) -> tensor<3x1x2xf32>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3x1x2xf32>, tensor<3x1x2xf32>) -> ()
    "func.return"(%4) : (tensor<3x1x2xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<7x5x3xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0xCAD97FC08B003B3F77D5DD3FC6D3933FE7AB7740DD3D62407498623F35F8B840B0486EBF82690440752943BFD0F6B6C0D6DE8540F4728A3FA38E15407B66B53FE057AEC09DA6C03E42DFA93FEB711540100412C17FB0FE401998C33D89C005C1545106C0E1BE03BFB51F96BF72E0CFBF2134BBBF6F2078BEBD18324052AE87BD596DB040480263BF4D238C405EABF13EBA083340D4F9953DBB38343F31877CC0B695B3BFCF879FC0FADF05403BD1503F7B2E2C3F1A88943FD51FAA3FBE6A2AC05F9D21BFBF5CC6406CDC613F36CEDE3E4A8493BFED7A07C03D2CA2BF60D6A6BF035D733F130C394063930F400AFEC1BF2E721F40355095BFC41E4FBF42F01CC0FC3F72C033CC4040F5A18D3FD4021CC0F01926C0C9A8B94085E391BF564D4540382C0FBFE0A717C0686A9BC0053ED9BFB8B9F5BFB9CD303EB588AA3F302930C02E9B90C084AC743E05E5CFBF1000E53F68FB38C09FA8973FA40A28C06B594FBF6B2C84C0E57EBABFE3519C3F1AB2733F3E03943EC2D84EC0C832D4BF324087C0943F0CC0984C72C0DE02D23FC0B8F9BD2F4F04C0DE2C65C0ACB60DBE5F8631BE7FF4C7BF"> : tensor<7x5x3xf32>}> : () -> tensor<7x5x3xf32>
    "func.return"(%1) : (tensor<7x5x3xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x1x2xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[-1.16651022, -0.809063196]], [[-1.91973019, 0.17265977]], [[0.951936364, 0.289087236]]]> : tensor<3x1x2xf32>}> : () -> tensor<3x1x2xf32>
    "func.return"(%0) : (tensor<3x1x2xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

