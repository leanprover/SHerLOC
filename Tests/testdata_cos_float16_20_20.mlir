"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<20x20xf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<20x20xf16>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<20x20xf16>
    %4 = "stablehlo.cosine"(%2) : (tensor<20x20xf16>) -> tensor<20x20xf16>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<20x20xf16>, tensor<20x20xf16>) -> ()
    "func.return"(%4) : (tensor<20x20xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x41446440F1462DC4F2C068BC613DF84096BDB03E423F3BBE10934E3A404096C392BBDABD78C03FC444C23835E0401845033DA2BD45C4D43A7DC3CCC0313CAFC31BC2444167BAE4C10CBEFB436DC51BC8F6431DC5C94152BE12C31FBD1CB6BCBE8F397AC093425A3ED442324526428CC1C5B632B38FB85BBDBCBB88413E3B43C2DD4474C6BEC2143BE2C3A73DCAB84C40DC422DBC07494EC3172A4A388A3A86C535BF47C690C56A47DABF7EC06EC10A392FBC1AC21AC9853AC2C6CBBCECBCD8C187C155C4E2443441D1B4B84453C229B9B7389EC75D96AA4000BA293F0CC02D41523DB9C4DD422F4329AF2E447DC1BBC080BCD8C001B8EEC0BE3E82C02FBCDBC5A4370C3F1EC2D9385E444C3C60BA74BB75C102C593BEB539BC4141C8AAC02725E03D3A48E9BEC2349E454EC03445A7BC04C086BED942493D993C6BC141C379BE0342A73BAD3DFB4003B889416A41EABB3D415D4635BFECBEFD411E393840F5B91EBD862C734589BA6A4000C1E4C00BC5EDBD50C8E03DF3414242C8409645B2C3F6C4DEC39AC491BBA5C1663CA4BCF9465B3F953AD9C27D3D6D2800420AB2C9BE13446EC73DACE3BE7D4461BF7EAE11409D3EEB42CE42BEC5B4C0A8406CC12D362E402BC0FD3D35C505464CC48BC1CFAC0EC0813C46C1AB34D8BB9EC2F24319C071C304453BC1FB406E4043B8E3BCDE3BFA3C9BC579BCC6BFA435C43A0CC1A8986544EABF8BBB833C8FC205B92BBAEEBCC4C448B626B80C3C15B41FC69544813E0F43B7B84344C7C02644A5C21AB8343DC7C095BED4BE53447ABEABC310C49EC45940A24295AF91BD0BC43AC36BBEAFBCC63D2DC447B81CC30CC4E2440A441836AAAC233E07C4C3C01A314FBCBEBE6A419ABD6F3FE1C4984446B9C2B9C2BE764334BA19BD20C3AAC5703260C6A5C152460C44884092BA8CBE32425FC3683CADC299C2853D743CE8360E3AC541DF42823A6AC0AC42D5C3FABC0040B9BE8FB0A73C313AAC3E39334BC6433F0E419341F6BB64C5ADABBCC2153B7544144425BD893E8DC2FFC32A43E23A4FC102412A3D5ABCF0C507C7043B174624BE55BBFA43684159BCC0BE3FB6CEC2D3BC9B3DCA415AB75FB9CBBCFDC6AF3F45C3E9C35CC57BC4"> : tensor<20x20xf16>}> : () -> tensor<20x20xf16>
    "func.return"(%1) : (tensor<20x20xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x15B7ADB8543A17B847BA3C372C3355BA8D3175AEB8B3BF22003CA43936B85DBAAD38E42EEDB831B700BC943B19BAF43500352E31DBB6423997BAE4B9FE371EBAF8BBFDBA9239D9BB8F2B4AB93C3997B559B93E36C1BBC1A063BB95346D3B34AF253AF4B8EABB60A4B4BB6E37FBBB77BB4B3BCC3BBC3A5B338B3871BBF13800BCCF30E23BCABB113993B907319C3A5EB8ACBB063877BAFBBAFE3BE13A7939CD3953B3003C033AD0361DB600B948BB773A0238F8BBB2B97D391E3BD3355735CFBB6FBBF1B56D31DDBAA43B831E00BC653AA73A8233003C85B9DA39F6B2FFB6CEBAA1334221AABB34BBF33B10B860BBB5B9E63604BA053B3DBA54AF0DB90238483B1B3B13B2F9BB933A6AB59F379739C63854BBA934A7AC0D3AB3BBDEB885B9003C852E83B8FFB0A63B4A3A65B88A375836C6B6AFABAFBBE7338B3643BB13BB10AAEDBB9C38D7305DBA043B72BB42BB6538EFBAF93B53B316B1EABB6B3A1AB8E2399934FB3B60397A39C1B869BA23BA3235B62D93B9852EE4BB00BCD9B9223A16BADF339FB914AFAE3898BB433763362C3A3DB47139AFBB5132FF3BECBBDC3B01B0C4B89636FC3BCFB01DB354B4F53B23B747AD9ABBBBBBDC3AA1B97FB945BB693BF0B7DBB7B72C9837B93B75B675BBFA3B0EB7E23600BBA93B7438E4BB65B95CB7B2BAC734EBBA5DBACEB8E43A79356F3822353B3AFF36D3B5823B4E3985BA003C00B558B6B338DB36EDBB7A3ABD395035D02A643BF33A3E38BE3BE53B29B00FAB68BBA73AF8B6D6B947B8DFBBF93A4534D6B9C7AC59B00EB630AA28BAD7B815AE8AB8E1BBF23BB431F7B820BB50A83B36113017B8E23A54BBF1B86D31FDB86D3BFB3BB02810B9CBB9E63B953754AF42BB6D318AB44E3193AF533A043A94AFA7BAB739AC344DBB843AD73BF73B98BBFF3BF1B81FB9733937ACFEBBD8BA3C37D9BBE7BB12321137443BD139BDBBA8BB7F39C1B8D9BBB8B92235A9B604AFEB3B5836B93936AECC3B003CC0B38ABA81BB5B380539FC3BCCBB11390BB4BEB87E3407ACEEBB3EB93DBB383912BB6EBA6B346E37873BE3391E39DA3B9028DE384DB93EBB723774AF663BBBBBB5356531C2BB2C3B443AD335183A7DB50CBB7FB9D3385BB3"> : tensor<20x20xf16>}> : () -> tensor<20x20xf16>
    "func.return"(%0) : (tensor<20x20xf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

